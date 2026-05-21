use ndarray::{Array2};
use ndarray_linalg::{Eigh, UPLO};

fn test_eigh(a: &Array2<f32>) {
    let _ = a.eigh(UPLO::Upper);
}
fn main() {}
