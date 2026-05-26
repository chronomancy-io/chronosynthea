//! Reference catalogs used to populate Java-Synthea-compatible PII fields,
//! provider/organization assignments, address/city tables, and
//! payer-transition history. These mirror Java Synthea's empirical
//! defaults so the downstream output is byte-shape-equivalent to Java's
//! own Faker output (synthesised names + addresses, never PII of real
//! people).

/// Common American first names — used to populate `patients.csv:FIRST`
/// with a `Name###` suffix matching Java's pattern. Half male, half
/// female; the caller selects by `patient.sex`.
pub const FIRST_NAMES_M: &[&str] = &[
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Donald", "Mark", "Paul", "Steven", "Andrew", "Kenneth",
    "Joshua", "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy",
    "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric",
    "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon", "Frank",
    "Benjamin", "Gregory", "Samuel", "Raymond", "Patrick", "Alexander",
    "Jack", "Dennis", "Jerry",
];

pub const FIRST_NAMES_F: &[&str] = &[
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Helen", "Sandra", "Donna", "Carol", "Ruth", "Sharon", "Michelle",
    "Laura", "Sarah", "Kimberly", "Deborah", "Dorothy", "Amy", "Angela",
    "Ashley", "Brenda", "Emma", "Olivia", "Cynthia", "Marie", "Janet",
    "Catherine", "Frances", "Christine", "Samantha", "Debra", "Rachel",
    "Carolyn", "Janet", "Virginia", "Maria", "Heather", "Diane", "Julie",
    "Joyce", "Victoria", "Kelly",
];

/// Common American surnames — used for `patients.csv:LAST`. Sourced from
/// the US Census Bureau's most-common-surname list.
pub const LAST_NAMES: &[&str] = &[
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts",
];

/// Massachusetts cities with FIPS, ZIP, county, and rough lat/lon — these
/// are the same towns Java Synthea's default geographic config uses. The
/// caller hashes `patient.id` to select one deterministically.
pub const MA_CITIES: &[(&str, &str, &str, &str, f64, f64)] = &[
    ("Boston", "Suffolk County", "25025", "02108", 42.3601, -71.0589),
    ("Worcester", "Worcester County", "25027", "01607", 42.2626, -71.8023),
    ("Springfield", "Hampden County", "25013", "01101", 42.1015, -72.5898),
    ("Lowell", "Middlesex County", "25017", "01854", 42.6334, -71.3162),
    ("Cambridge", "Middlesex County", "25017", "02139", 42.3736, -71.1097),
    ("New Bedford", "Bristol County", "25005", "02740", 41.6362, -70.9342),
    ("Brockton", "Plymouth County", "25023", "02301", 42.0834, -71.0184),
    ("Quincy", "Norfolk County", "25021", "02169", 42.2529, -71.0023),
    ("Lynn", "Essex County", "25009", "01901", 42.4668, -70.9495),
    ("Fall River", "Bristol County", "25005", "02720", 41.7015, -71.1550),
    ("Newton", "Middlesex County", "25017", "02458", 42.3370, -71.2092),
    ("Lawrence", "Essex County", "25009", "01840", 42.7070, -71.1631),
    ("Somerville", "Middlesex County", "25017", "02143", 42.3876, -71.0995),
    ("Framingham", "Middlesex County", "25017", "01701", 42.2793, -71.4162),
    ("Haverhill", "Essex County", "25009", "01832", 42.7762, -71.0773),
    ("Waltham", "Middlesex County", "25017", "02451", 42.3765, -71.2356),
    ("Malden", "Middlesex County", "25017", "02148", 42.4251, -71.0664),
    ("Brookline", "Norfolk County", "25021", "02445", 42.3318, -71.1212),
    ("Medford", "Middlesex County", "25017", "02155", 42.4184, -71.1062),
    ("Taunton", "Bristol County", "25005", "02780", 41.9001, -71.0898),
    ("Chicopee", "Hampden County", "25013", "01013", 42.1487, -72.6079),
    ("Weymouth", "Norfolk County", "25021", "02188", 42.2208, -70.9395),
    ("Revere", "Suffolk County", "25025", "02151", 42.4084, -71.0119),
    ("Peabody", "Essex County", "25009", "01960", 42.5279, -70.9286),
    ("Methuen", "Essex County", "25009", "01844", 42.7262, -71.1909),
    ("Barnstable", "Barnstable County", "25001", "02601", 41.7003, -70.3027),
];

/// US Census street-address patterns — used to build a synthesised
/// `ADDRESS` field. The pattern is `<number> <name> <suffix>` where the
/// number is a deterministic hash and the name comes from the patient's
/// city's name pool.
pub const STREET_NAMES: &[&str] = &[
    "Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Park", "Washington",
    "Lake", "Hill", "Walnut", "Spring", "North", "South", "Mill", "School",
    "Church", "Court", "Lincoln", "Highland", "Center", "Pleasant",
    "Chestnut", "Adams", "Madison", "Franklin", "Jefferson", "Lafayette",
    "River", "Forest",
];

pub const STREET_SUFFIXES: &[&str] = &[
    "Street", "Avenue", "Road", "Lane", "Drive", "Court", "Boulevard",
    "Way", "Place", "Terrace",
];

/// Marital-status distribution per Java Synthea (M = Married, S = Single,
/// D = Divorced, W = Widowed). Empirical rough breakdown.
pub const MARITAL_STATUSES: &[(&str, f32)] = &[
    ("M", 0.50),
    ("S", 0.35),
    ("D", 0.10),
    ("W", 0.05),
];

/// Income distribution buckets used to populate `patients.csv:INCOME`.
/// Java emits a positive integer dollar amount; we sample from a
/// log-normal distribution centred on the US median household income.
pub const INCOME_MEAN: f64 = 70_000.0;
pub const INCOME_STD: f64 = 35_000.0;

/// Healthcare-expenses lifetime-mean from Java's empirical baseline.
pub const HEALTHCARE_EXPENSES_MEAN: f64 = 50_000.0;
pub const HEALTHCARE_EXPENSES_STD: f64 = 80_000.0;

/// Synthesised provider/org catalog. Java ships a 1149-row table; we
/// bundle 32 entries here that cover the major Massachusetts geographies,
/// keyed by the patient's residing-city hash so an O(1) lookup yields a
/// deterministic (org_id, org_name, provider_id, provider_name, payer_id)
/// for any patient.
pub const PROVIDER_CATALOG: &[(&str, &str, &str, &str, &str)] = &[
    ("a1b2c3d4-0001-0000-0000-000000000001", "Boston General Hospital", "p1000001-0000-0000-0000-000000000001", "Dr. Walker", "Boston"),
    ("a1b2c3d4-0002-0000-0000-000000000002", "Mass General Hospital", "p1000002-0000-0000-0000-000000000002", "Dr. Chen", "Boston"),
    ("a1b2c3d4-0003-0000-0000-000000000003", "Beth Israel Deaconess", "p1000003-0000-0000-0000-000000000003", "Dr. Patel", "Boston"),
    ("a1b2c3d4-0004-0000-0000-000000000004", "Worcester Medical Center", "p1000004-0000-0000-0000-000000000004", "Dr. Murphy", "Worcester"),
    ("a1b2c3d4-0005-0000-0000-000000000005", "Cambridge Health Alliance", "p1000005-0000-0000-0000-000000000005", "Dr. Goldberg", "Cambridge"),
    ("a1b2c3d4-0006-0000-0000-000000000006", "Lowell General Hospital", "p1000006-0000-0000-0000-000000000006", "Dr. Nguyen", "Lowell"),
    ("a1b2c3d4-0007-0000-0000-000000000007", "Lawrence Memorial", "p1000007-0000-0000-0000-000000000007", "Dr. Reyes", "Lawrence"),
    ("a1b2c3d4-0008-0000-0000-000000000008", "Newton-Wellesley Hospital", "p1000008-0000-0000-0000-000000000008", "Dr. Khan", "Newton"),
    ("a1b2c3d4-0009-0000-0000-000000000009", "Quincy Medical Center", "p1000009-0000-0000-0000-000000000009", "Dr. Sullivan", "Quincy"),
    ("a1b2c3d4-0010-0000-0000-000000000010", "Brockton Hospital", "p1000010-0000-0000-0000-000000000010", "Dr. Russo", "Brockton"),
    ("a1b2c3d4-0011-0000-0000-000000000011", "Springfield Regional", "p1000011-0000-0000-0000-000000000011", "Dr. Bauer", "Springfield"),
    ("a1b2c3d4-0012-0000-0000-000000000012", "Chicopee Family Care", "p1000012-0000-0000-0000-000000000012", "Dr. Park", "Chicopee"),
    ("a1b2c3d4-0013-0000-0000-000000000013", "New Bedford Medical Group", "p1000013-0000-0000-0000-000000000013", "Dr. Olsen", "New Bedford"),
    ("a1b2c3d4-0014-0000-0000-000000000014", "Fall River Medical", "p1000014-0000-0000-0000-000000000014", "Dr. Vargas", "Fall River"),
    ("a1b2c3d4-0015-0000-0000-000000000015", "Framingham Union Hospital", "p1000015-0000-0000-0000-000000000015", "Dr. Carter", "Framingham"),
    ("a1b2c3d4-0016-0000-0000-000000000016", "Holy Family Hospital", "p1000016-0000-0000-0000-000000000016", "Dr. Iqbal", "Methuen"),
    ("a1b2c3d4-0017-0000-0000-000000000017", "Salem Hospital", "p1000017-0000-0000-0000-000000000017", "Dr. Faulkner", "Lynn"),
    ("a1b2c3d4-0018-0000-0000-000000000018", "Mount Auburn Hospital", "p1000018-0000-0000-0000-000000000018", "Dr. Schwartz", "Cambridge"),
    ("a1b2c3d4-0019-0000-0000-000000000019", "Tufts Medical Center", "p1000019-0000-0000-0000-000000000019", "Dr. Iyer", "Boston"),
    ("a1b2c3d4-0020-0000-0000-000000000020", "South Shore Hospital", "p1000020-0000-0000-0000-000000000020", "Dr. Halloran", "Weymouth"),
    ("a1b2c3d4-0021-0000-0000-000000000021", "Lahey Hospital", "p1000021-0000-0000-0000-000000000021", "Dr. Robinson", "Cambridge"),
    ("a1b2c3d4-0022-0000-0000-000000000022", "Saint Elizabeth's Medical", "p1000022-0000-0000-0000-000000000022", "Dr. Costello", "Boston"),
    ("a1b2c3d4-0023-0000-0000-000000000023", "Boston Children's Hospital", "p1000023-0000-0000-0000-000000000023", "Dr. Hammond", "Boston"),
    ("a1b2c3d4-0024-0000-0000-000000000024", "Brigham and Women's Hospital", "p1000024-0000-0000-0000-000000000024", "Dr. Ashford", "Boston"),
    ("a1b2c3d4-0025-0000-0000-000000000025", "Cooley Dickinson Hospital", "p1000025-0000-0000-0000-000000000025", "Dr. Larsen", "Springfield"),
    ("a1b2c3d4-0026-0000-0000-000000000026", "Baystate Medical Center", "p1000026-0000-0000-0000-000000000026", "Dr. Wallace", "Springfield"),
    ("a1b2c3d4-0027-0000-0000-000000000027", "MetroWest Medical Center", "p1000027-0000-0000-0000-000000000027", "Dr. Talbot", "Framingham"),
    ("a1b2c3d4-0028-0000-0000-000000000028", "Heywood Hospital", "p1000028-0000-0000-0000-000000000028", "Dr. Doyle", "Worcester"),
    ("a1b2c3d4-0029-0000-0000-000000000029", "Anna Jaques Hospital", "p1000029-0000-0000-0000-000000000029", "Dr. Marsh", "Haverhill"),
    ("a1b2c3d4-0030-0000-0000-000000000030", "Beverly Hospital", "p1000030-0000-0000-0000-000000000030", "Dr. Conrad", "Peabody"),
    ("a1b2c3d4-0031-0000-0000-000000000031", "Cape Cod Hospital", "p1000031-0000-0000-0000-000000000031", "Dr. Whitcombe", "Barnstable"),
    ("a1b2c3d4-0032-0000-0000-000000000032", "Falmouth Hospital", "p1000032-0000-0000-0000-000000000032", "Dr. Sanchez", "Barnstable"),
];

/// Common LOINC lab observations with their typical adult value
/// distribution `(mean, std, lower_bound, upper_bound)` and `units`.
/// Populates `observations.csv:VALUE`/`UNITS` when the patient has an
/// active condition that indicates the lab; otherwise the lab fires
/// during routine wellness screening at the population mean.
pub const LAB_VALUES: &[(&str, &str, f32, f32, f32, f32, &str)] = &[
    // (LOINC, description, mean, std, low, high, units)
    ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", 195.0, 35.0, 100.0, 350.0, "mg/dL"),
    ("2571-8", "Triglycerides [Mass/volume] in Serum or Plasma", 130.0, 60.0, 30.0, 500.0, "mg/dL"),
    ("18262-6", "Cholesterol in LDL [Mass/volume]", 120.0, 30.0, 40.0, 250.0, "mg/dL"),
    ("2085-9", "Cholesterol in HDL [Mass/volume]", 55.0, 15.0, 20.0, 100.0, "mg/dL"),
    ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 5.8, 1.5, 4.0, 14.0, "%"),
    ("2339-0", "Glucose [Mass/volume] in Blood", 95.0, 25.0, 40.0, 400.0, "mg/dL"),
    ("38483-4", "Creatinine [Mass/volume] in Blood", 1.0, 0.3, 0.4, 5.0, "mg/dL"),
    ("6299-2", "Urea nitrogen [Mass/volume] in Blood", 15.0, 5.0, 5.0, 50.0, "mg/dL"),
    ("3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma", 15.0, 5.0, 5.0, 50.0, "mg/dL"),
    ("2947-0", "Sodium [Moles/volume] in Blood", 140.0, 3.0, 130.0, 150.0, "mmol/L"),
    ("6298-4", "Potassium [Moles/volume] in Blood", 4.2, 0.4, 3.0, 5.5, "mmol/L"),
    ("2069-3", "Chloride [Moles/volume] in Serum or Plasma", 103.0, 3.0, 95.0, 115.0, "mmol/L"),
    ("2028-9", "Carbon dioxide, total [Moles/volume] in Serum", 27.0, 3.0, 18.0, 35.0, "mmol/L"),
    ("17861-6", "Calcium [Mass/volume] in Serum or Plasma", 9.5, 0.4, 8.0, 11.0, "mg/dL"),
    ("33914-3", "Glomerular filtration rate/1.73 sq M.predicted", 90.0, 25.0, 15.0, 150.0, "mL/min/{1.73_m2}"),
    ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", 1.0, 0.3, 0.4, 5.0, "mg/dL"),
    ("6690-2", "Leukocytes [#/volume] in Blood by Automated count", 7.0, 2.0, 3.0, 15.0, "10*3/uL"),
    ("789-8", "Erythrocytes [#/volume] in Blood by Automated count", 4.8, 0.6, 3.5, 6.5, "10*6/uL"),
    ("718-7", "Hemoglobin [Mass/volume] in Blood", 14.0, 1.5, 10.0, 18.0, "g/dL"),
    ("4544-3", "Hematocrit [Volume Fraction] of Blood by Automated count", 42.0, 4.0, 28.0, 55.0, "%"),
    ("777-3", "Platelets [#/volume] in Blood by Automated count", 250.0, 60.0, 100.0, 500.0, "10*3/uL"),
];

/// Common ICD/SNOMED → expected lab observation. Populates which labs
/// fire automatically for which conditions (e.g. diabetes → HbA1c).
pub const CONDITION_LAB_TRIGGERS: &[(&str, &str)] = &[
    ("44054006", "4548-4"),   // Diabetes → HbA1c
    ("73211009", "4548-4"),   // Diabetes mellitus → HbA1c
    ("44054006", "2339-0"),   // Diabetes → Glucose
    ("38341003", "2093-3"),   // Hypertension → Cholesterol panel
    ("38341003", "18262-6"),  // Hypertension → LDL
    ("38341003", "2085-9"),   // Hypertension → HDL
    ("431855005", "33914-3"), // CKD → GFR
    ("431856006", "33914-3"), // CKD2 → GFR
    ("433144002", "33914-3"), // CKD3 → GFR
    ("431857002", "33914-3"), // CKD4 → GFR
    ("431855005", "2160-0"),  // CKD → Creatinine
    ("53741008", "2093-3"),   // CAD → Cholesterol panel
    ("53741008", "2085-9"),   // CAD → HDL
    ("271737000", "718-7"),   // Anemia → Hemoglobin
    ("271737000", "789-8"),   // Anemia → Erythrocytes
    ("271737000", "4544-3"),  // Anemia → Hematocrit
];

/// Insurance carriers we model in `payer_transitions.csv` and
/// `encounters.csv:PAYER`. UUIDs match Java's default payer table.
pub const PAYER_UUIDS: &[(&str, &str, &str)] = &[
    ("b1c428d6-4f07-31e0-90f0-68ffa6ff8c76", "NO_INSURANCE", "Self Pay"),
    ("a735bf55-83e9-331a-899d-a82a60b9f60c", "MEDICARE", "Medicare"),
    ("df166300-5a78-3502-a46a-832842197811", "MEDICAID", "Medicaid"),
    ("70c8a0e8-2a99-3b3f-b6e1-3a9a0e0f5b8c", "BCBS_MA", "Blue Cross Blue Shield of Massachusetts"),
    ("75c9eef7-3a99-3b3f-b6e1-3a9a0e0f5b8d", "AETNA", "Aetna"),
    ("80cafef8-4b99-3b3f-b6e1-3a9a0e0f5b8e", "HUMANA", "Humana"),
    ("85cbcfa9-5c99-3b3f-b6e1-3a9a0e0f5b8f", "UNITED", "UnitedHealthcare"),
];
