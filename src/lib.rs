use pyo3::prelude::*;

mod rust_impl;

#[pyfunction]
fn hello_from_bin() -> String {
    "RUST: Hello from LitData!".to_string()
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    m.add_class::<rust_impl::fs::s3::S3Storage>()?;
    Ok(())
}
