#![allow(warnings)]
pub use num_traits::{Float, Num, NumAssign, One, Zero};
use std::array;
use std::cmp::{Ordering, PartialOrd};
use std::iter::{Extend, IntoIterator};
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, Sub, SubAssign};

use crate::traits::{CloneFloat, CloneNum, CloneNumAssign, Pow, PowAssign};

trait PowiCustom {
    fn powi_custom(&self, n: i32) -> Self;
}

impl<T: CloneNum> PowiCustom for T {
    fn powi_custom(&self, n: i32) -> Self {
        let mut result = T::one();
        let mut base = self.clone();
        let mut exponent = n;

        while exponent > 0 {
            if exponent % 2 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            exponent = exponent / 2;
        }

        result
    }
}

// Mathematical Vector

#[derive(PartialEq, Eq, Ord, Hash, Clone, Copy, Debug)]
pub struct Vector<const N: usize = 2, T: CloneNum = i32> {
    value: [T; N],
}

#[macro_export]
macro_rules! vector {
    [$($e:expr),+ $(,)?] => {
        {
            let mut values = vec![$($e),+];
            Vector::new(values.try_into().unwrap())
        }
    };
}

impl<T: CloneNum, const N: usize> Vector<N, T> {
    pub const fn new(value: [T; N]) -> Self {
        Self { value }
    }

    pub fn norm_sq(&self) -> T {
        let mut result = T::zero();
        for i in self.value.clone() {
            result = result + i.clone() * i.clone();
        }
        result
    }

    pub fn dot_prod(self, rhs: Self) -> T {
        let mut result = T::zero();
        for i in 0..N {
            result = result.clone() + self[i].clone() * rhs[i].clone();
        }
        result
    }

    pub const fn dim(&self) -> usize {
        N
    }
}

impl<T: CloneNum + Neg<Output = T>, const N: usize> Vector<N, T> {
    fn cross_prod_ex(args: &Vec<Self>) -> Self {
        let mut re = Self::default();
        let mut a: Matrix<N, N, T> = Matrix::zero();
        for i in 1..N {
            a[i] = args[i - 1].clone().value;
        }
        for i in 0..N - 1 {
            let mut a = a.clone();
            a[0][i] = T::one();
            re[i] = a.det();
        }
        re
    }
}

impl<T: CloneFloat, const N: usize> Vector<N, T> {
    pub fn norm(&self) -> T {
        self.norm_sq().sqrt()
    }
}

impl<T: CloneNum + Neg<Output = T>, const N: usize> Neg for Vector<N, T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut x = self.clone();

        for i in &mut x {
            *i = -i.clone();
        }
        x
    }
}

impl<T: CloneNum, const N: usize> Add for Vector<N, T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for i in 0..N {
            result[i] = result[i].clone() + rhs[i].clone();
        }
        result
    }
}

impl<T: CloneNumAssign, const N: usize> AddAssign for Vector<N, T> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self[i] += rhs[i].clone();
        }
    }
}

impl<T: CloneNum, const N: usize> Sub for Vector<N, T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut x = self.clone();

        for i in 0..N {
            x[i] = x[i].clone() - rhs[i].clone();
        }
        x
    }
}

impl<T: CloneNumAssign, const N: usize> SubAssign for Vector<N, T> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self[i] -= rhs[i].clone();
        }
    }
}

impl<T: CloneNum, const N: usize> Mul<T> for Vector<N, T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut x = self.clone();

        for i in &mut x {
            *i = i.clone() * rhs.clone();
        }
        x
    }
}

impl<T: CloneNumAssign, const N: usize> MulAssign<T> for Vector<N, T> {
    fn mul_assign(&mut self, rhs: T) {
        for i in self {
            *i *= rhs.clone();
        }
    }
}

impl<T: CloneNum, const N: usize> Mul<Self> for Vector<N, T> {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot_prod(rhs)
    }
}

impl<T: CloneNum + Neg<Output = T>> Vector<3, T> {
    pub fn cross_prod(&self, rhs: Self) -> Self {
        Self::new([
            Matrix::new([
                [T::one(), T::zero(), T::zero()],
                self.value.clone(),
                rhs.value.clone(),
            ])
            .det(),
            Matrix::new([
                [T::zero(), T::one(), T::zero()],
                self.value.clone(),
                rhs.value.clone(),
            ])
            .det(),
            Matrix::new([
                [T::zero(), T::zero(), T::one()],
                self.value.clone(),
                rhs.value.clone(),
            ])
            .det(),
        ])
    }
}

impl<T: CloneNumAssign, const N: usize> MulAssign<Self> for Vector<N, T> {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self[i] *= rhs[i].clone();
        }
    }
}

impl<T: CloneNum, const N: usize> Zero for Vector<N, T> {
    fn zero() -> Self {
        Self::new(array::from_fn(|_| T::zero().clone()))
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl<T: CloneNum + PartialOrd, const N: usize> PartialOrd for Vector<N, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_value = self.norm_sq();
        let other_value = other.norm_sq();
        if self_value == other_value {
            Some(Ordering::Equal)
        } else if self_value > other_value {
            Some(Ordering::Greater)
        } else if self_value < other_value {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

impl<T: CloneNum, const N: usize> IntoIterator for Vector<N, T> {
    type Item = T;
    type IntoIter = <[T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.value.into_iter()
    }
}

impl<'a, T: CloneNum, const N: usize> IntoIterator for &'a Vector<N, T> {
    type Item = &'a T;
    type IntoIter = <&'a [T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.value).into_iter()
    }
}

impl<'a, T: CloneNum, const N: usize> IntoIterator for &'a mut Vector<N, T> {
    type Item = <&'a mut [T; N] as IntoIterator>::Item;
    type IntoIter = <&'a mut [T; N] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.value).into_iter()
    }
}

impl<T: CloneNum, I, const N: usize> Index<I> for Vector<N, T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &(&self.value as &[T])[index]
    }
}

impl<T: CloneNum, I, const N: usize> IndexMut<I> for Vector<N, T>
where
    [T]: IndexMut<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut (&mut self.value as &mut [T])[index]
    }
}

impl<T: CloneNum + std::fmt::Debug, const N: usize> TryFrom<Vec<T>> for Vector<N, T> {
    type Error = <[T; N] as TryFrom<Vec<T>>>::Error;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        match value.try_into() {
            Ok(x) => Ok(Vector::new(x)),
            Err(e) => Err(e),
        }
    }
}

impl<T: CloneNum, const N: usize> Default for Vector<N, T> {
    fn default() -> Self {
        Self::zero()
    }
}

// Matrix

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct Matrix<const H: usize = 2, const W: usize = 2, T: CloneNum = i32> {
    value: [[T; W]; H],
}

impl<T: CloneNum, const H: usize, const W: usize> Matrix<H, W, T> {
    pub const fn new(value: [[T; W]; H]) -> Self {
        Self { value }
    }

    pub const fn is_reg(&self) -> bool {
        H == W
    }

    pub const fn size(&self) -> usize {
        H * W
    }

    pub fn row(&self, h: usize) -> Matrix<1, W, T> {
        Matrix::new([self[h].clone()])
    }

    pub fn col(&self, w: usize) -> Matrix<H, 1, T> {
        let mut x = array::from_fn(|_| [T::zero()]);
        for i in 0..W {
            x[i][0] = self.get_item(i, w).clone();
        }
        Matrix::new(x)
    }

    pub fn row_vec(&self, h: usize) -> Vector<W, T> {
        Vector::new(self[h].clone())
    }

    pub fn col_vec(&self, w: usize) -> Vector<H, T> {
        let mut x = Vector::<H, T>::zero();
        for i in 0..H {
            x[i] = self[i][w].clone();
        }
        x
    }

    pub fn get_item(&self, h: usize, w: usize) -> T {
        self.value[h][w].clone()
    }

    pub fn get_item_ref(&self, h: usize, w: usize) -> &T {
        &self.value[h][w]
    }

    pub fn get_item_mut(&mut self, h: usize, w: usize) -> &mut T {
        &mut self.value[h][w]
    }

    pub fn to_vec(self) -> Vec<Vec<T>> {
        let mut x = Vec::new();
        for i in self {
            x.push(i.to_vec());
        }
        x
    }
}

impl<T: CloneNum + Neg<Output = T>, const N: usize> Matrix<N, N, T> {
    pub fn det(&self) -> T {
        Self::det_c(self.clone().to_vec(), N)
    }

    fn det_c(t: Vec<Vec<T>>, n: usize) -> T {
        if n == 1 {
            return t[0][0].clone();
        }

        let next = n - 1;
        let mut sign = if n % 2 == 1 { T::one() } else { -T::one() };
        let mut r = T::zero();
        let mut l = t.clone();
        for w in 0..n {
            l[w].pop();
        }
        for i in 0..n {
            let mut l_clone = l.clone();
            l_clone.remove(i);
            r = r + sign.clone() * t[i][n - 1].clone() * Self::det_c(l_clone, next);
            sign = -sign;
        }

        r
    }

    fn minor(&self, h: usize, w: usize) -> T {
        let mut v = self.clone().to_vec();
        v.remove(h);
        for i in 0..N - 1 {
            v[i].remove(w);
        }
        Self::det_c(v, N)
    }

    pub fn cof(&self, h: usize, w: usize) -> T {
        (if h + w % 2 == 0 { T::one() } else { -T::one() }) * self.minor(h, w)
    }

    //    pub fn inv(&self) -> Self {
    //        let det = self.det();
    //        let mut re = Self::default();
    //        for i in 0..N {
    //            for j in 0..N {
    //                re[i][j] =
    //            }
    //        }
    //    }
}

impl<T: CloneNum, I, const H: usize, const W: usize> Index<I> for Matrix<H, W, T>
where
    [[T; W]]: Index<I>,
{
    type Output = <[[T; W]] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &(&self.value as &[[T; W]])[index]
    }
}

impl<T: CloneNum, I, const H: usize, const W: usize> IndexMut<I> for Matrix<H, W, T>
where
    [[T; W]]: IndexMut<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut (&mut self.value as &mut [[T; W]])[index]
    }
}

impl<T: CloneNum, const H: usize, const W: usize> IntoIterator for Matrix<H, W, T> {
    type Item = <[[T; W]; H] as IntoIterator>::Item;
    type IntoIter = <[[T; W]; H] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.value.into_iter()
    }
}

impl<'a, T: CloneNum, const H: usize, const W: usize> IntoIterator for &'a Matrix<H, W, T> {
    type Item = <&'a [[T; W]; H] as IntoIterator>::Item;
    type IntoIter = <&'a [[T; W]; H] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.value).into_iter()
    }
}

impl<'a, T: CloneNum, const H: usize, const W: usize> IntoIterator for &'a mut Matrix<H, W, T> {
    type Item = <&'a mut [[T; W]; H] as IntoIterator>::Item;
    type IntoIter = <&'a mut [[T; W]; H] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&mut self.value).into_iter()
    }
}

impl<T: CloneNum, const N: usize> From<Vector<N, T>> for Matrix<1, N, T> {
    fn from(value: Vector<N, T>) -> Self {
        Matrix::new([value.value])
    }
}

impl<T: CloneNum + Neg<Output = T>, const H: usize, const W: usize> Neg for Matrix<H, W, T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut re = self.clone();
        for i in &mut re {
            for j in i {
                *j = -(j.clone());
            }
        }
        re
    }
}

impl<T: CloneNum, const H: usize, const W: usize> Add for Matrix<H, W, T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut x = Self::zero();
        for i in 0..H {
            for j in 0..W {
                x[i][j] = self[i][j].clone() + rhs[i][j].clone();
            }
        }
        x
    }
}

impl<T: CloneNumAssign, const H: usize, const W: usize> AddAssign for Matrix<H, W, T> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..H {
            for j in 0..W {
                self[i][j] += rhs[i][j].clone();
            }
        }
    }
}

impl<T: CloneNum, const H: usize, const W: usize> Sub for Matrix<H, W, T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut x = Self::zero();
        for i in 0..H {
            for j in 0..W {
                x[i][j] = self[i][j].clone() + rhs[i][j].clone();
            }
        }
        x
    }
}

impl<T: CloneNumAssign, const H: usize, const W: usize> SubAssign for Matrix<H, W, T> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..H {
            for j in 0..W {
                self[i][j] -= rhs[i][j].clone();
            }
        }
    }
}

impl<T: CloneNum, const H: usize, const W: usize> Mul<T> for Matrix<H, W, T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut x = Self::zero();
        for i in 0..H {
            for j in 0..W {
                x[i][j] = rhs.clone() * self[i][j].clone();
            }
        }
        x
    }
}

impl<T: CloneNumAssign, const H: usize, const W: usize> MulAssign<T> for Matrix<H, W, T> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..H {
            for j in 0..W {
                self[i][j] *= rhs.clone();
            }
        }
    }
}

impl<T: CloneNum, const H: usize, const W: usize, const P: usize> Mul<Matrix<W, P, T>>
    for Matrix<H, W, T>
{
    type Output = Matrix<H, P, T>;

    fn mul(self, rhs: Matrix<W, P, T>) -> Self::Output {
        let mut x = Self::Output::zero();
        for i in 0..H {
            for j in 0..P {
                let mut r = T::zero();
                for k in 0..W {
                    r = r.clone() + self[i][k].clone() * rhs[k][j].clone();
                }
                x[i][j] = r;
            }
        }
        x
    }
}

impl<T: CloneNumAssign, const H: usize> MulAssign for Matrix<H, H, T> {
    fn mul_assign(&mut self, rhs: Self) {
        let mut x = Self::zero();
        for i in 0..H {
            for j in 0..H {
                let mut r = T::zero();
                for k in 0..H {
                    r += self[i][k].clone() * rhs[k][j].clone();
                }
                self[i][j] = r;
            }
        }
    }
}

impl<T: CloneNum, U: CloneNum, const H: usize> Pow<U> for Matrix<H, H, T> {
    type Output = Self;

    fn pow(self, rhs: U) -> Self::Output {
        let two = U::one() + U::one();
        if rhs == U::zero() {
            Self::one()
        } else if rhs.clone() % two.clone() == U::one() {
            self.clone() * self.pow(rhs - U::one())
        } else {
            let a = self.pow(rhs / two);
            a.clone() * a.clone()
        }
    }
}

impl<T: CloneNumAssign, U: CloneNum, const H: usize> PowAssign<U> for Matrix<H, H, T> {
    fn pow_assign(&mut self, rhs: U) {
        *self = self.clone().pow(rhs);
    }
}

impl<T: CloneNum, const H: usize, const W: usize> Zero for Matrix<H, W, T> {
    fn zero() -> Self {
        Self::new(array::from_fn(|_| array::from_fn(|_| T::zero())))
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl<T: CloneNum, const N: usize> One for Matrix<N, N, T> {
    fn one() -> Self {
        let mut x = Self::zero();
        for i in 0..N {
            x[i][i] = T::one();
        }
        x
    }
}

impl<T: CloneNum, const H: usize, const W: usize> Default for Matrix<H, W, T> {
    fn default() -> Self {
        Self::zero()
    }
}
