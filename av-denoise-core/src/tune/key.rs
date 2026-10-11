use std::fmt::{self, Display};

/// Separates the fields of a site key's text form.
const FIELD_SEPARATOR: char = '|';

/// A site and every compile-time value its kernels are built for.
///
/// Width and height are rounded up to the next power of two, so nearby resolutions share a key.
/// Every other field is exact. Values derived only from width and height are covered by the two
/// buckets and are left out. The text form is stable, because stored results are keyed by it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SiteKey(String);

impl SiteKey {
    pub fn builder(site: &'static str, width: u32, height: u32) -> SiteKeyBuilder {
        let width_bucket = bucket_dim(width);
        let height_bucket = bucket_dim(height);
        let text = format!("{site}{FIELD_SEPARATOR}w{width_bucket}{FIELD_SEPARATOR}h{height_bucket}");

        SiteKeyBuilder { text }
    }

    /// Restores a key from its stored text, rejecting text with no site name.
    pub fn from_stored(text: &str) -> Option<Self> {
        let site = text.split(FIELD_SEPARATOR).next()?;
        if site.is_empty() {
            return None;
        }

        let owned = text.to_string();
        Some(Self(owned))
    }

    /// The site id, the first field of the key.
    pub fn site(&self) -> &str {
        self.0.split(FIELD_SEPARATOR).next().unwrap_or_default()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for SiteKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Builds a [SiteKey](crate::tune::SiteKey) one field at a time.
#[derive(Debug, Clone)]
pub struct SiteKeyBuilder {
    text: String,
}

impl SiteKeyBuilder {
    /// Appends a `name=value` field.
    pub fn param(mut self, name: &str, value: impl Display) -> Self {
        let field = format!("{FIELD_SEPARATOR}{name}={value}");
        self.text.push_str(&field);
        self
    }

    pub fn build(self) -> SiteKey {
        SiteKey(self.text)
    }
}

/// Rounds a frame dimension up to the next power of two.
pub fn bucket_dim(dim: u32) -> u32 {
    dim.next_power_of_two()
}
