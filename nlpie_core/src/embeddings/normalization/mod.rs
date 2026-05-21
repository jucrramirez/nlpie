pub mod centering;
pub mod error;
pub mod preprocessor;
pub mod standardization;
pub mod utils;
pub mod whitening;

// Re-export common structs and functions to maintain the same API surface.
pub use centering::{mean_center, FitStats};
pub use error::PreprocessingError;
pub use preprocessor::EmbeddingPreprocessor;
pub use standardization::standardize_columns;
pub use utils::{cosine_similarity, l2_normalize_rows, DEFAULT_EPS};
pub use whitening::{remove_top_principal_components, whiten_pca, WhitenModel};
