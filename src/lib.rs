/* 
 * Free FFT and convolution (Rust)
 * 
 * Copyright (c) 2017 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/free-small-fft-in-multiple-languages
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */

#![no_std]

extern crate containers;

use core::iter::repeat;
use containers::collections::Vec;
use cplx::Complex;


/* 
 * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
 * The vector can have any length. This is a wrapper function.
 */
pub fn transform(xs: &mut [Complex<f64>]) {
	let n: usize = xs.len();
	if n == 0 {
		return;
	} else if n & (n - 1) == 0 {  // Is power of 2
		transform_radix2(xs);
	} else {  // More complicated algorithm for arbitrary sizes
		transform_bluestein(xs);
	}
}


/* 
 * Computes the inverse discrete Fourier transform (IDFT) of the given complex vector, storing the result back into the vector.
 * The vector can have any length. This is a wrapper function. This transform does not perform scaling, so the inverse is not a true inverse.
 */
pub fn inverse_transform(xs: &mut [Complex<f64>]) {
        for x in xs.iter_mut() { let (a, b) = x.to_rect(); *x = Complex::from_rect(b, a) }
	transform(xs);
        for x in xs.iter_mut() { let (a, b) = x.to_rect(); *x = Complex::from_rect(b, a) }
}

mod c_math {
    #[link(name = "m")]
    extern {
        pub fn sin(_: f64) -> f64;
        pub fn cos(_: f64) -> f64;
    }
}

fn sin(x: f64) -> f64 { unsafe { c_math::sin(x) } }
fn cos(x: f64) -> f64 { unsafe { c_math::cos(x) } }

/* 
 * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
 * The vector's length must be a power of 2. Uses the Cooley-Tukey decimation-in-time radix-2 algorithm.
 */
pub fn transform_radix2(xs: &mut [Complex<f64>]) {
	// Length variables
	let n: usize = real.len();
	if n != imag.len() {
		panic!("Mismatched lengths");
	}
	let levels: u32 = n.trailing_zeros();
	if 1usize << levels != n {
		panic!("Length is not a power of 2");
	}
	
	// Trigonometric tables
	let mut costable: Vec<f64> = Vec::with_capacity(n / 2).unwrap();
	let mut sintable: Vec<f64> = Vec::with_capacity(n / 2).unwrap();
	for i in 0 .. n / 2 {
		let angle: f64 = 2.0 * core::f64::consts::PI * (i as f64) / (n as f64);
		costable.push(cos(angle)).unwrap();
		sintable.push(sin(angle)).unwrap();
	}
	
	fn reverse_bits(mut x: usize, n: u32) -> usize {
		let mut result: usize = 0;
		for _ in 0 .. n {
			result = (result << 1) | (x & 1);
			x >>= 1;
		}
		result
	}
	
	// Bit-reversed addressing permutation
	for i in 0 .. n {
		let j: usize = reverse_bits(i, levels);
		if j > i {
			real.swap(i, j);
			imag.swap(i, j);
		}
	}
	
	// Cooley-Tukey decimation-in-time radix-2 FFT
	let mut size: usize = 2;
	while size <= n {
		let halfsize: usize = size / 2;
		let tablestep: usize = n / size;
		let mut i = 0;
		while i < n {
			let mut k: usize = 0;
			for j in i .. i + halfsize {
				let l: usize = j + halfsize;
				let tpre: f64 =  real[l] * costable[k] + imag[l] * sintable[k];
				let tpim: f64 = -real[l] * sintable[k] + imag[l] * costable[k];
				real[l] = real[j] - tpre;
				imag[l] = imag[j] - tpim;
				real[j] += tpre;
				imag[j] += tpim;
				k += tablestep;
			}
			i += size;
		}
		if size == n {  // Prevent overflow in 'size *= 2'
			break;
		}
		size *= 2;
	}
}


/* 
 * Computes the discrete Fourier transform (DFT) of the given complex vector, storing the result back into the vector.
 * The vector can have any length. This requires the convolution function, which in turn requires the radix-2 FFT function.
 * Uses Bluestein's chirp z-transform algorithm.
 */
pub fn transform_bluestein(real: &mut [f64], imag: &mut [f64]) {
	// Find a power-of-2 convolution length m such that m >= n * 2 + 1
	let n: usize = real.len();
	if n != imag.len() {
		panic!("Mismatched lengths");
	}
	let mut m: usize = 1;
	while m / 2 <= n {
		match m.checked_mul(2) {
			None => { panic!("Array too large"); },
			Some(newm) => { m = newm; },
		}
	}
	
	// Trignometric tables
	let mut costable: Vec<f64> = Vec::with_capacity(n).unwrap();
	let mut sintable: Vec<f64> = Vec::with_capacity(n).unwrap();
	for i in 0 .. n {
		  // This is more accurate than j = i * i
		let j: u64 = (i as u64) * (i as u64) % ((n as u64) * 2);
		let angle: f64 = core::f64::consts::PI * (j as f64) / (n as f64);
		costable.push(cos(angle)).unwrap();
		sintable.push(sin(angle)).unwrap();
	}
	
	// Temporary vectors and preprocessing
	let mut areal: Vec<f64> = Vec::with_capacity(m).unwrap();
	let mut aimag: Vec<f64> = Vec::with_capacity(m).unwrap();
	for i in 0 .. n {
		areal.push( real[i] * costable[i] + imag[i] * sintable[i]).unwrap();
		aimag.push(-real[i] * sintable[i] + imag[i] * costable[i]).unwrap();
	}
	for _ in n .. m {
		areal.push(0.0).unwrap();
		aimag.push(0.0).unwrap();
	}
	let mut breal: Vec<f64> = vec_of(0.0, m).unwrap();
	let mut bimag: Vec<f64> = vec_of(0.0, m).unwrap();
	breal[0] = costable[0];
	bimag[0] = sintable[0];
	for i in 1 .. n {
		breal[i] = costable[i];
		breal[m - i] = costable[i];
		bimag[i] = sintable[i];
		bimag[m - i] = sintable[i];
	}
	
	// Convolution
	let mut creal: Vec<f64> = vec_of(0.0, m).unwrap();
	let mut cimag: Vec<f64> = vec_of(0.0, m).unwrap();
	convolve_complex(&areal, &aimag, &breal, &bimag, &mut creal, &mut cimag);
	
	// Postprocessing
	for i in 0 .. n {
		real[i] =  creal[i] * costable[i] + cimag[i] * sintable[i];
		imag[i] = -creal[i] * sintable[i] + cimag[i] * costable[i];
	}
}


/* 
 * Computes the circular convolution of the given real vectors. Each vector's length must be the same.
 */
pub fn convolve_real(x: &[f64], y: &[f64], out: &mut [f64]) {
	let n: usize = x.len();
	if n != y.len() || n != out.len() {
		panic!("Mismatched lengths");
	}
	convolve_complex(x, &mut vec_of(0.0, n).unwrap(), y, &mut vec_of(0.0, n).unwrap(), out, &mut vec_of(0.0, n).unwrap());
}


/* 
 * Computes the circular convolution of the given complex vectors. Each vector's length must be the same.
 */
pub fn convolve_complex(xreal: &[f64], ximag: &[f64], yreal: &[f64], yimag: &[f64],
		outreal: &mut [f64], outimag: &mut [f64]) {
	
	let n: usize = xreal.len();
	if n != ximag.len() || n != yreal.len() || n != yimag.len()
			|| n != outreal.len() || n != outimag.len() {
		panic!("Mismatched lengths");
	}
	
	let mut xrecp: Vec<f64> = Vec::from_iter(xreal.iter().cloned()).unwrap();
	let mut ximcp: Vec<f64> = Vec::from_iter(ximag.iter().cloned()).unwrap();
	let mut yrecp: Vec<f64> = Vec::from_iter(yreal.iter().cloned()).unwrap();
	let mut yimcp: Vec<f64> = Vec::from_iter(yimag.iter().cloned()).unwrap();
	transform(&mut xrecp, &mut ximcp);
	transform(&mut yrecp, &mut yimcp);
	
	for i in 0 .. n {
		let temp: f64 = xrecp[i] * yrecp[i] - ximcp[i] * yimcp[i];
		ximcp[i] = ximcp[i] * yrecp[i] + xrecp[i] * yimcp[i];
		xrecp[i] = temp;
	}
	inverse_transform(&mut xrecp, &mut ximcp);
	
	for i in 0 .. n {  // Scaling (because this FFT implementation omits it)
		outreal[i] = xrecp[i] / (n as f64);
		outimag[i] = ximcp[i] / (n as f64);
	}
}

fn vec_of<T: Clone>(x: T, n: usize) -> Option<Vec<T>> { Vec::from_iter(repeat(x).take(n)).ok() }
