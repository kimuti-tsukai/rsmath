use num_traits::{CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, Float, Num, NumAssign};

pub trait CloneNum: Num + Clone {}

pub trait CloneNumAssign: CloneNum + NumAssign {}

pub trait CloneFloat: Float + CloneNum {}

pub trait CheckedOps: CheckedAdd + CheckedSub + CheckedMul + CheckedDiv {}

pub trait CheckedNum: Num + CheckedOps {}

impl<T: Num + Clone> CloneNum for T {}
impl<T: CloneNum + NumAssign> CloneNumAssign for T {}
impl<T: Float + CloneNum> CloneFloat for T {}
impl<T: CheckedAdd + CheckedSub + CheckedMul + CheckedDiv> CheckedOps for T {}
impl<T: Num + CheckedOps> CheckedNum for T {}

pub trait Pow<T: CloneNum = Self> {
    type Output;

    fn pow(self, rhs: T) -> Self::Output;
}

pub trait PowAssign<T: CloneNum = Self> {
    fn pow_assign(&mut self, rhs: T);
}

macro_rules! impl_pow_unsigned {
    ($($t: ty),*) => {
        $(
            impl Pow for $t {
                type Output = Self;

                fn pow(self, rhs: $t) -> Self::Output {
                    let mut r = 1;

                    for _ in 0..rhs {
                        r *= self;
                    }

                    r
                }
            }
        )*
    };
}

macro_rules! impl_pow_assign_unsigned {
    ($($t: ty),*) => {
        $(
            impl PowAssign for $t {
                fn pow_assign(&mut self, rhs: $t) {
                    let mut r = 1;

                    for _ in 0..rhs {
                        r *= *self;
                    }

                    *self = r;
                }
            }
        )*
    };
}

macro_rules! impl_pow_signed {
    ($($t: ty),*) => {
        $(
            impl Pow for $t {
                type Output = Self;

                fn pow(self, rhs: $t) -> Self::Output {
                    let mut r = 1;

                    for _ in 0..rhs.abs() {
                        r *= self;
                    }

                    if rhs >= 0 {
                        r
                    } else {
                        1 / r
                    }
                }
            }
        )*
    };
}

macro_rules! impl_pow_assign_signed {
    ($($t: ty),*) => {
        $(
            impl PowAssign for $t {
                fn pow_assign(&mut self, rhs: $t) {
                    let mut r = 1;

                    for _ in 0..rhs.abs() {
                        r *= *self;
                    }

                    *self = if rhs >= 0 {
                        r
                    } else {
                        1 / r
                    };
                }
            }
        )*
    };
}

impl_pow_unsigned!(u8, u16, u32, u64, u128, usize);

impl_pow_assign_unsigned!(u8, u16, u32, u64, u128, usize);

impl_pow_signed!(i8, i16, i32, i64, i128, isize);

impl_pow_assign_signed!(i8, i16, i32, i64, i128, isize);
