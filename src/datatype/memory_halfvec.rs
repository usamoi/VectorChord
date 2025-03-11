use half::f16;
use pgrx::datum::{FromDatum, IntoDatum};
use pgrx::pg_sys::{Datum, Oid};
use pgrx::pgrx_sql_entity_graph::metadata::*;
use std::marker::PhantomData;
use std::ptr::NonNull;
use vector::VectorBorrowed;
use vector::vect::VectBorrowed;

#[repr(C, align(8))]
struct HalfvecHeader {
    varlena: u32,
    dims: u16,
    unused: u16,
    elements: [f16; 0],
}

impl HalfvecHeader {
    fn size_of(len: usize) -> usize {
        if len > 65535 {
            panic!("vector is too large");
        }
        (size_of::<Self>() + size_of::<f16>() * len).next_multiple_of(8)
    }
    unsafe fn as_borrowed<'a>(this: NonNull<Self>) -> VectBorrowed<'a, f16> {
        unsafe {
            let this = this.as_ptr();
            VectBorrowed::new(std::slice::from_raw_parts(
                (&raw const (*this).elements).cast(),
                (&raw const (*this).dims).read() as usize,
            ))
        }
    }
}

pub struct HalfvecInput<'a>(NonNull<HalfvecHeader>, PhantomData<&'a ()>, bool);

impl HalfvecInput<'_> {
    unsafe fn from_ptr(p: NonNull<HalfvecHeader>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum(p.as_ptr().cast()).cast()).unwrap()
        };
        HalfvecInput(q, PhantomData, p != q)
    }
    pub fn as_borrowed(&self) -> VectBorrowed<'_, f16> {
        unsafe { HalfvecHeader::as_borrowed(self.0) }
    }
}

impl Drop for HalfvecInput<'_> {
    fn drop(&mut self) {
        if self.2 {
            unsafe {
                pgrx::pg_sys::pfree(self.0.as_ptr().cast());
            }
        }
    }
}

pub struct HalfvecOutput(NonNull<HalfvecHeader>);

impl HalfvecOutput {
    unsafe fn from_ptr(p: NonNull<HalfvecHeader>) -> Self {
        let q = unsafe {
            NonNull::new(pgrx::pg_sys::pg_detoast_datum_copy(p.as_ptr().cast()).cast()).unwrap()
        };
        Self(q)
    }
    #[allow(dead_code)]
    pub fn new(vector: VectBorrowed<'_, f16>) -> Self {
        unsafe {
            let slice = vector.slice();
            let size = HalfvecHeader::size_of(slice.len());

            let ptr = pgrx::pg_sys::palloc0(size) as *mut HalfvecHeader;
            (&raw mut (*ptr).varlena).write((size << 2) as u32);
            (&raw mut (*ptr).dims).write(vector.dims() as _);
            (&raw mut (*ptr).unused).write(0);
            std::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                (&raw mut (*ptr).elements).cast(),
                slice.len(),
            );
            Self(NonNull::new(ptr).unwrap())
        }
    }
    pub fn as_borrowed(&self) -> VectBorrowed<'_, f16> {
        unsafe { HalfvecHeader::as_borrowed(self.0) }
    }
    fn into_raw(self) -> *mut HalfvecHeader {
        let result = self.0.as_ptr();
        std::mem::forget(self);
        result
    }
}

impl Drop for HalfvecOutput {
    fn drop(&mut self) {
        unsafe {
            pgrx::pg_sys::pfree(self.0.as_ptr().cast());
        }
    }
}

// FromDatum

impl FromDatum for HalfvecInput<'_> {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
            unsafe { Some(Self::from_ptr(ptr)) }
        }
    }
}

impl FromDatum for HalfvecOutput {
    unsafe fn from_polymorphic_datum(datum: Datum, is_null: bool, _typoid: Oid) -> Option<Self> {
        if is_null {
            None
        } else {
            let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
            unsafe { Some(Self::from_ptr(ptr)) }
        }
    }
}

// IntoDatum

impl IntoDatum for HalfvecOutput {
    fn into_datum(self) -> Option<Datum> {
        Some(Datum::from(self.into_raw()))
    }

    fn type_oid() -> Oid {
        Oid::INVALID
    }

    fn is_compatible_with(_: Oid) -> bool {
        true
    }
}

// UnboxDatum

unsafe impl<'a> pgrx::datum::UnboxDatum for HalfvecInput<'a> {
    type As<'src>
        = HalfvecInput<'src>
    where
        'a: 'src;
    #[inline]
    unsafe fn unbox<'src>(datum: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let datum = datum.sans_lifetime();
        let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
        unsafe { Self::from_ptr(ptr) }
    }
}

unsafe impl pgrx::datum::UnboxDatum for HalfvecOutput {
    type As<'src> = HalfvecOutput;
    #[inline]
    unsafe fn unbox<'src>(datum: pgrx::datum::Datum<'src>) -> Self::As<'src>
    where
        Self: 'src,
    {
        let datum = datum.sans_lifetime();
        let ptr = NonNull::new(datum.cast_mut_ptr()).unwrap();
        unsafe { Self::from_ptr(ptr) }
    }
}

// SqlTranslatable

unsafe impl SqlTranslatable for HalfvecInput<'_> {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("halfvec")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("halfvec"))))
    }
}

unsafe impl SqlTranslatable for HalfvecOutput {
    fn argument_sql() -> Result<SqlMapping, ArgumentError> {
        Ok(SqlMapping::As(String::from("halfvec")))
    }
    fn return_sql() -> Result<Returns, ReturnsError> {
        Ok(Returns::One(SqlMapping::As(String::from("halfvec"))))
    }
}

// ArgAbi

unsafe impl<'fcx> pgrx::callconv::ArgAbi<'fcx> for HalfvecInput<'fcx> {
    unsafe fn unbox_arg_unchecked(arg: pgrx::callconv::Arg<'_, 'fcx>) -> Self {
        let index = arg.index();
        unsafe {
            arg.unbox_arg_using_from_datum()
                .unwrap_or_else(|| panic!("argument {index} must not be null"))
        }
    }
}

// BoxRet

unsafe impl pgrx::callconv::BoxRet for HalfvecOutput {
    unsafe fn box_into<'fcx>(
        self,
        fcinfo: &mut pgrx::callconv::FcInfo<'fcx>,
    ) -> pgrx::datum::Datum<'fcx> {
        match self.into_datum() {
            Some(datum) => unsafe { fcinfo.return_raw_datum(datum) },
            None => fcinfo.return_null(),
        }
    }
}
