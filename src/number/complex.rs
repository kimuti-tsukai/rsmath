#![allow(warnings)]
use crate::traits::{CloneFloat, CloneNum, CloneNumAssign};

use num_traits::{Num, One, Zero};

use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Not, Rem, RemAssign, Sub, SubAssign,
};

use std::fmt::{Debug, Display};

use crate::{impl_for_ref, impl_for_ref_assign};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Complex<T: CloneNum> {
    pub re: T,
    pub im: T,
}

impl<T: CloneNum> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    pub fn re(&self) -> T {
        self.re.clone()
    }

    pub fn im(&self) -> T {
        self.im.clone()
    }

    pub fn abs_sq(&self) -> T {
        self.re() * self.re() + self.im() * self.im()
    }
}

impl<T: CloneNum + Neg<Output = T>> Complex<T> {
    pub fn conj(&self) -> Self {
        Self::new(self.re(), -self.im())
    }
}

impl<T: CloneFloat> Complex<T> {
    pub fn abs(&self) -> T {
        self.abs_sq().sqrt()
    }
}

impl<T: CloneNum> Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re() + rhs.re(), self.im() + rhs.im())
    }
}

impl<T: CloneNumAssign> AddAssign for Complex<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re();
        self.im += rhs.im();
    }
}

impl<T: CloneNum> Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re() - rhs.re(), self.im() - rhs.im())
    }
}

impl<T: CloneNumAssign> SubAssign for Complex<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re();
        self.im -= rhs.im();
    }
}

impl<T: CloneNum> Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re() * rhs.re() - self.im() * rhs.im(),
            self.re() * rhs.im() + self.im() * rhs.re(),
        )
    }
}

impl<T: CloneNumAssign> MulAssign for Complex<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: CloneNum> Div for Complex<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let den = rhs.re() * rhs.re() + rhs.im() * rhs.im();
        Self::new(
            (self.re() * rhs.re() + self.im() * rhs.im()) / den.clone(),
            (self.im() * rhs.re() - self.re() * rhs.im()) / den.clone(),
        )
    }
}

impl<T: CloneNumAssign> DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<T: CloneNum> Rem for Complex<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let Complex { re, im } = self.clone() / rhs.clone();
        let gaussian = Complex::new(re.clone() - re % T::one(), im.clone() - im % T::one());

        self - rhs * gaussian
    }
}

impl<T: CloneNumAssign> RemAssign for Complex<T> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = self.clone() % rhs;
    }
}

type U<T> = Complex<T>;

impl_for_ref!(U<T>, Add, add);

impl_for_ref!(U<T>, Sub, sub);

impl_for_ref!(U<T>, Mul, mul);

impl_for_ref!(U<T>, Div, div);

impl_for_ref!(U<T>, Rem, rem);

impl_for_ref_assign!(U<T>, AddAssign, add_assign);

impl_for_ref_assign!(U<T>, SubAssign, sub_assign);

impl_for_ref_assign!(U<T>, MulAssign, mul_assign);

impl_for_ref_assign!(U<T>, DivAssign, div_assign);

impl_for_ref_assign!(U<T>, RemAssign, rem_assign);

impl<T: CloneNum> Zero for Complex<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }
}

impl<T: CloneNum> One for Complex<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }
}

impl<T: CloneNum> Default for Complex<T> {
    fn default() -> Self {
        Self::zero()
    }
}
