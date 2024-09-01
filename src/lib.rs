#![allow(warnings)]
pub mod linear_algebra;
use crate::linear_algebra::*;

pub mod traits;
use crate::traits::*;

pub mod number;
use crate::number::frac::*;

pub mod functions;
use crate::functions::*;

#[macro_use]
mod macros;
#[macro_use]
use crate::macros::*;

pub use num_traits::{One, Zero};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg() {
        let x = Vector::new([1, 2]);
        assert_eq!(-x, Vector::new([-1, -2]));
    }

    #[test]
    fn add_matrix() {
        assert_eq!(
            Matrix::new([[1, 2, 3], [1, 3, 5]]) + Matrix::new([[2, 4, 6], [1, 2, 3]]),
            Matrix::new([[3, 6, 9], [2, 5, 8]])
        );
    }

    #[test]
    fn vector_macro() {
        let x = vector![1, 2, 3];
        let y = Vector::new([1, 2, 3]);
        assert_eq!(x, y);
    }

    #[test]
    fn test_matrix_multiplication() {
        // 2x2行列の例
        let a = Matrix::new([[1, 2], [3, 4]]);
        let b = Matrix::new([[5, 6], [7, 8]]);
        let expected = Matrix::new([[19, 22], [43, 50]]);

        // 行列の掛け算
        let result = a * b;

        // 結果が期待値と一致するかを確認
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication_zero_matrix() {
        let a = Matrix::new([[0, 0], [0, 0]]);
        let b = Matrix::new([[1, 2], [3, 4]]);
        let expected = Matrix::new([[0, 0], [0, 0]]);

        let result = a * b;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_multiplication_identity_matrix() {
        let a = Matrix::new([[1, 2], [3, 4]]);
        let b = Matrix::one();
        let expected = a.clone();

        let result = a * b;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_det_square_matrix() {
        let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let det = matrix.det();
        assert_eq!(det, -2.0);
    }

    #[test]
    fn test_det_zero_determinant() {
        let matrix = Matrix::new([[1.0, 2.0], [2.0, 4.0]]);
        let det = matrix.det();
        assert_eq!(det, 0.0);
    }

    #[test]
    fn test_det_all_zero_matrix() {
        let matrix = Matrix::new([[0.0, 0.0], [0.0, 0.0]]);
        let det = matrix.det();
        assert_eq!(det, 0.0);
    }

    #[test]
    fn test_det_identity_matrix() {
        let matrix = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
        let det = matrix.det();
        assert_eq!(det, 1.0);
    }

    #[test]
    fn det_int() {
        let matrix = Matrix::new([[1, 2], [3, 4]]);
        let det = matrix.det();
        assert_eq!(det, -2);
    }

    #[test]
    fn frac_from_num() {
        Frac::from(10);
        let _: Frac<Frac<i32>> = Frac::from(Frac::new(1, 2));
    }

    #[test]
    fn for_stament_like_c() {
        let mut w = 0;
        for_c! {(i = 0; i < 10; i+=1) {
            w += i;
        }}
        assert_eq!(w, 45);
    }

    #[test]
    fn for_stament_const() {
        const X: usize = {
            let mut w = 0;
            for_c! {(i = 0; i < 10; i+=1) {
                w += i;
            }}
            w
        };
        assert_eq!(X, 45);
    }
}
