/// A family's name, unique within its site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FamilyId(pub &'static str);

/// The family a dimension site's baseline is the representative of.
pub const BASELINE_FAMILY: FamilyId = FamilyId("baseline");

/// How a site's candidates group into families.
#[derive(Debug, Clone, Copy)]
pub enum Families<C: 'static> {
    /// Disjoint families that together hold every candidate.
    Flat(&'static [FlatFamily<C>]),
    /// Candidates that are a cross product of dimensions, where each dimension's value is a family.
    Dimensions(DimensionFamilies<C>),
}

/// One family of a flat site.
#[derive(Debug, Clone, Copy)]
pub struct FlatFamily<C: 'static> {
    pub id: FamilyId,
    /// The member that stands in for the family.
    pub representative: C,
    /// Every member, the representative included.
    pub members: &'static [C],
}

/// The dimensions of a cross-product site, with how its candidates map to and from their values.
#[derive(Debug, Clone, Copy)]
pub struct DimensionFamilies<C: 'static> {
    pub dimensions: &'static [Dimension],
    /// The index of the candidate's value in the dimension at the given index.
    pub value_of: fn(C, usize) -> usize,
    /// The candidate with one value index per dimension, or `None` if the site does not build it.
    pub compose: fn(&[usize]) -> Option<C>,
}

/// One axis of a cross-product site.
#[derive(Debug, Clone, Copy)]
pub struct Dimension {
    pub name: &'static str,
    pub values: &'static [DimensionValue],
}

/// One value of a dimension, which is one family.
#[derive(Debug, Clone, Copy)]
pub struct DimensionValue {
    pub id: FamilyId,
}

/// Every family's representative, with the family it stands in for.
///
/// A dimension site's representatives are `baseline`, under [BASELINE_FAMILY], then each of the
/// other values with every other dimension at the baseline's value. Values whose representative the
/// site does not build are left out. Flat sites declare their own representatives.
pub fn representatives<C: Copy>(families: &Families<C>, baseline: C) -> Vec<(FamilyId, C)> {
    match families {
        Families::Flat(flat) => flat
            .iter()
            .map(|family| (family.id, family.representative))
            .collect(),
        Families::Dimensions(dimensions) => dimension_representatives(dimensions, baseline),
    }
}

fn dimension_representatives<C: Copy>(families: &DimensionFamilies<C>, baseline: C) -> Vec<(FamilyId, C)> {
    let base = base_values(families, baseline);
    let mut values = base.clone();
    let mut representatives = vec![(BASELINE_FAMILY, baseline)];

    for (dimension_index, dimension) in families.dimensions.iter().enumerate() {
        for (value_index, value) in dimension.values.iter().enumerate() {
            if value_index == base[dimension_index] {
                continue;
            }

            values[dimension_index] = value_index;

            if let Some(candidate) = (families.compose)(&values) {
                representatives.push((value.id, candidate));
            }
        }

        values[dimension_index] = base[dimension_index];
    }

    representatives
}

/// The baseline's value index in each dimension.
fn base_values<C: Copy>(families: &DimensionFamilies<C>, baseline: C) -> Vec<usize> {
    (0..families.dimensions.len())
        .map(|dimension| (families.value_of)(baseline, dimension))
        .collect()
}

/// Every candidate in the families named in `opened`.
///
/// A dimension site gives the cross product of each dimension's opened values, where the value
/// `baseline` takes always counts as opened. Combinations the site does not build are left out.
pub fn opened_members<C: Copy>(families: &Families<C>, baseline: C, opened: &[FamilyId]) -> Vec<C> {
    match families {
        Families::Flat(flat) => flat
            .iter()
            .filter(|family| opened.contains(&family.id))
            .flat_map(|family| family.members.iter().copied())
            .collect(),
        Families::Dimensions(dimensions) => cross_product(dimensions, baseline, opened),
    }
}

fn cross_product<C: Copy>(families: &DimensionFamilies<C>, baseline: C, opened: &[FamilyId]) -> Vec<C> {
    let base = base_values(families, baseline);
    let opened_values: Vec<Vec<usize>> = families
        .dimensions
        .iter()
        .zip(&base)
        .map(|(dimension, &base_index)| opened_value_indices(dimension, base_index, opened))
        .collect();

    let mut combinations: Vec<Vec<usize>> = vec![Vec::new()];
    for indices in &opened_values {
        combinations = combinations
            .iter()
            .flat_map(|prefix| extend_with(prefix, indices))
            .collect();
    }

    combinations
        .iter()
        .filter_map(|values| (families.compose)(values))
        .collect()
}

/// The indices of the dimension's values that are opened, `base_index` always among them.
fn opened_value_indices(dimension: &Dimension, base_index: usize, opened: &[FamilyId]) -> Vec<usize> {
    dimension
        .values
        .iter()
        .enumerate()
        .filter(|&(index, value)| index == base_index || opened.contains(&value.id))
        .map(|(index, _)| index)
        .collect()
}

fn extend_with(prefix: &[usize], indices: &[usize]) -> Vec<Vec<usize>> {
    indices
        .iter()
        .map(|&index| {
            let mut combination = prefix.to_vec();
            combination.push(index);
            combination
        })
        .collect()
}
