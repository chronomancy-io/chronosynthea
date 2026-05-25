//! Newtype wrappers for the MSS pipeline's index types.
//!
//! `ArchetypeId` and `ConditionIndex` are both `u16` underneath but are
//! distinct types so the compiler can reject confusions of one for the
//! other at API boundaries — e.g. passing an archetype id where
//! `ConditionIndex` was expected. This matches the
//! [C-NEWTYPE Rust API Guideline](https://rust-lang.github.io/api-guidelines/type-safety.html#newtypes-encapsulate-implementation-details-c-newtype)
//! and the Wave 2 council's `MCP rust-best-practices` audit recommendation.
//!
//! Both newtypes are `#[repr(transparent)]` so they have identical memory
//! layout to a bare `u16` — zero runtime or storage cost.

use serde::{Deserialize, Serialize};

/// Identifier of a patient archetype within an `ArchetypeRegistry`.
///
/// Constructed by `ArchetypeRegistry::from_fingerprint`. Use the `From<u16>`
/// / `From<ArchetypeId>` impls or the public field for explicit conversion
/// at deserialisation or test-data boundaries.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ArchetypeId(pub u16);

impl From<u16> for ArchetypeId {
    #[inline]
    fn from(v: u16) -> Self {
        Self(v)
    }
}

impl From<ArchetypeId> for u16 {
    #[inline]
    fn from(v: ArchetypeId) -> Self {
        v.0
    }
}

impl ArchetypeId {
    /// Returns the underlying `u16` value.
    #[inline]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    /// Returns the underlying `u16` as a `usize` for slice indexing.
    #[inline]
    pub const fn as_index(self) -> usize {
        self.0 as usize
    }
}

/// Index of a single condition within the global condition table.
///
/// Distinct from [`ArchetypeId`] so the compiler refuses to swap them.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ConditionIndex(pub u16);

impl From<u16> for ConditionIndex {
    #[inline]
    fn from(v: u16) -> Self {
        Self(v)
    }
}

impl From<ConditionIndex> for u16 {
    #[inline]
    fn from(v: ConditionIndex) -> Self {
        v.0
    }
}

impl ConditionIndex {
    #[inline]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn as_index(self) -> usize {
        self.0 as usize
    }
}
