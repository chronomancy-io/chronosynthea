//! Streaming output for patient data.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;

use chronosynthea_core::Patient;

use crate::error::IoResult;
use crate::format::OutputFormat;

/// Buffer size for file I/O (8 MB for better throughput).
const BUFFER_SIZE: usize = 8 * 1024 * 1024;

/// Streaming writer for patient data.
pub struct StreamWriter<W: Write> {
    writer: BufWriter<W>,
    format: OutputFormat,
    count: usize,
    first: bool,
}

impl<W: Write> StreamWriter<W> {
    /// Creates a new stream writer.
    pub fn new(writer: W, format: OutputFormat) -> Self {
        Self {
            writer: BufWriter::with_capacity(BUFFER_SIZE, writer),
            format,
            count: 0,
            first: true,
        }
    }

    /// Writes a patient in the configured format.
    pub fn write(&mut self, patient: &Patient) -> IoResult<()> {
        match self.format {
            OutputFormat::Jsonl => {
                simd_json::serde::to_writer(&mut self.writer, patient)?;
                self.writer.write_all(b"\n")?;
            }
            OutputFormat::Compact => {
                let compact = patient.to_compact();
                simd_json::serde::to_writer(&mut self.writer, &compact)?;
                self.writer.write_all(b"\n")?;
            }
            OutputFormat::CodesOnly => {
                let codes = patient.to_codes_only();
                simd_json::serde::to_writer(&mut self.writer, &codes)?;
                self.writer.write_all(b"\n")?;
            }
            OutputFormat::Json => {
                if self.first {
                    self.writer.write_all(b"[\n")?;
                    self.first = false;
                } else {
                    self.writer.write_all(b",\n")?;
                }
                simd_json::serde::to_writer(&mut self.writer, patient)?;
            }
            OutputFormat::MessagePack => {
                // MessagePack binary format - much faster than JSON
                rmp_serde::encode::write(&mut self.writer, patient)?;
            }
        }
        self.count += 1;
        Ok(())
    }

    /// Finishes writing and returns the underlying writer.
    pub fn finish(mut self) -> IoResult<W> {
        if self.format == OutputFormat::Json && self.count > 0 {
            self.writer.write_all(b"\n]")?;
        }
        self.writer.flush()?;
        Ok(self.writer.into_inner().map_err(|e| e.into_error())?)
    }

    /// Returns the number of patients written.
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Creates a stream writer to a file.
///
/// Automatically detects gzip compression from the file extension.
pub fn create_file_writer(
    path: &Path,
    format: OutputFormat,
) -> IoResult<StreamWriter<Box<dyn Write>>> {
    let file = File::create(path)?;

    let writer: Box<dyn Write> = if path.extension().map(|e| e == "gz").unwrap_or(false) {
        Box::new(GzEncoder::new(file, Compression::default()))
    } else {
        Box::new(file)
    };

    Ok(StreamWriter::new(writer, format))
}

/// Writes patients to a file.
pub fn write_patients_to_file(
    path: &Path,
    patients: &[Patient],
    format: OutputFormat,
) -> IoResult<usize> {
    let mut writer = create_file_writer(path, format)?;

    for patient in patients {
        writer.write(patient)?;
    }

    writer.finish()?;
    Ok(patients.len())
}

/// Writes patients to a file with optimized sequential I/O.
///
/// Uses large buffers and pre-allocated scratch space for fast serialization.
pub fn write_patients_parallel(
    path: &Path,
    patients: &[Patient],
    format: OutputFormat,
) -> IoResult<usize> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, file);

    // Pre-allocate a scratch buffer for serialization
    let mut scratch = Vec::with_capacity(4096);

    if format == OutputFormat::Json {
        writer.write_all(b"[\n")?;
    }

    for (i, patient) in patients.iter().enumerate() {
        scratch.clear();

        match format {
            OutputFormat::Jsonl => {
                simd_json::serde::to_writer(&mut scratch, patient)?;
                writer.write_all(&scratch)?;
                writer.write_all(b"\n")?;
            }
            OutputFormat::Compact => {
                let compact = patient.to_compact();
                simd_json::serde::to_writer(&mut scratch, &compact)?;
                writer.write_all(&scratch)?;
                writer.write_all(b"\n")?;
            }
            OutputFormat::CodesOnly => {
                let codes = patient.to_codes_only();
                simd_json::serde::to_writer(&mut scratch, &codes)?;
                writer.write_all(&scratch)?;
                writer.write_all(b"\n")?;
            }
            OutputFormat::Json => {
                if i > 0 {
                    writer.write_all(b",\n")?;
                }
                simd_json::serde::to_writer(&mut scratch, patient)?;
                writer.write_all(&scratch)?;
            }
            OutputFormat::MessagePack => {
                // MessagePack binary format - much faster than JSON
                rmp_serde::encode::write(&mut scratch, patient)?;
                writer.write_all(&scratch)?;
            }
        }
    }

    if format == OutputFormat::Json {
        writer.write_all(b"\n]")?;
    }

    writer.flush()?;
    Ok(patients.len())
}

/// Progress-tracking stream writer wrapper.
pub struct ProgressStreamWriter<W: Write> {
    inner: StreamWriter<W>,
    report_interval: usize,
    callback: Box<dyn Fn(usize)>,
}

impl<W: Write> ProgressStreamWriter<W> {
    /// Creates a new progress-tracking writer.
    pub fn new(
        writer: W,
        format: OutputFormat,
        report_interval: usize,
        callback: impl Fn(usize) + 'static,
    ) -> Self {
        Self {
            inner: StreamWriter::new(writer, format),
            report_interval,
            callback: Box::new(callback),
        }
    }

    /// Writes a patient and reports progress.
    pub fn write(&mut self, patient: &Patient) -> IoResult<()> {
        self.inner.write(patient)?;
        if self.inner.count() % self.report_interval == 0 {
            (self.callback)(self.inner.count());
        }
        Ok(())
    }

    /// Finishes writing.
    pub fn finish(self) -> IoResult<W> {
        self.inner.finish()
    }

    /// Returns the count.
    pub fn count(&self) -> usize {
        self.inner.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, TimeZone, Utc};
    use chronosynthea_core::{Ethnicity, Race, Sex};

    fn create_test_patient() -> Patient {
        Patient {
            id: "P00000001".to_string(),
            birth_date: NaiveDate::from_ymd_opt(1990, 5, 15).unwrap(),
            sex: Sex::Male,
            race: Race::White,
            ethnicity: Ethnicity::NonHispanic,
            encounters: vec![],
        }
    }

    #[test]
    fn test_stream_writer_jsonl() {
        let mut buffer = Vec::new();
        let mut writer = StreamWriter::new(&mut buffer, OutputFormat::Jsonl);

        let patient = create_test_patient();
        writer.write(&patient).unwrap();
        writer.finish().unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("P00000001"));
        assert!(output.ends_with("\n"));
    }

    #[test]
    fn test_stream_writer_compact() {
        let mut buffer = Vec::new();
        let mut writer = StreamWriter::new(&mut buffer, OutputFormat::Compact);

        let patient = create_test_patient();
        writer.write(&patient).unwrap();
        writer.finish().unwrap();

        let output = String::from_utf8(buffer).unwrap();
        // Compact format uses short field names
        assert!(output.contains("\"bd\""));
        assert!(output.contains("\"s\""));
    }

    #[test]
    fn test_stream_writer_json() {
        let mut buffer = Vec::new();
        let mut writer = StreamWriter::new(&mut buffer, OutputFormat::Json);

        let patient = create_test_patient();
        writer.write(&patient).unwrap();
        writer.write(&patient).unwrap();
        writer.finish().unwrap();

        let output = String::from_utf8(buffer).unwrap();
        assert!(output.starts_with("["));
        assert!(output.ends_with("]"));
    }
}
