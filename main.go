package main

import (
	"fmt"
	"math/rand/v2"

	. "github.com/ArvindAnchi/n_back_prop/matrix"
)

type NN struct {
	a0 *Mat

	c  float32
	cd float32
	fp float32

	w1, b1, nw1, nb1, a1 *Mat
	w2, b2, nw2, nb2, a2 *Mat
}

const LR = 2e-1

func (m *NN) forward(inf bool) {
	m.a1.Dot(m.a0, m.w1)
	m.a1.Sum(m.b1)
	m.a1.Sigmoid()

	m.a2.Dot(m.a1, m.w2)
	m.a2.Sum(m.b2)
	m.a2.Sigmoid()

	if !inf {
		if m.cd < 1 {
			m.w1.Nudge(m.nw1, rand.Float32()*LR)
			m.b1.Nudge(m.nb1, rand.Float32()*LR)

			m.w2.Nudge(m.nw2, rand.Float32()*LR)
			m.b2.Nudge(m.nb2, rand.Float32()*LR)

			m.nw1.Flip(m.fp)
			m.nb1.Flip(m.fp)

			m.nw2.Flip(m.fp)
			m.nb2.Flip(m.fp)

			if m.fp > 0.1 {
				m.fp -= m.cd
			}
		}
	}
}

func (m *NN) cost(ti Mat, to Mat) float32 {
	if ti.Rows != to.Rows {
		panic(fmt.Sprintf("M_COST: Expected shape:(%d %d) got (%d %d)", to.Rows, m.a2.Cols, ti.Rows, ti.Cols))
	}
	if to.Cols != m.a2.Cols {
		panic(fmt.Sprintf("M_COST: Expected shape:(%d %d) got (%d %d)", to.Rows, m.a2.Cols, to.Rows, to.Cols))
	}

	n := ti.Rows
	var ci float32 = 0

	for i := 0; i < n; i++ {
		x := ti.Row(i)
		y := to.Row(i)

		m.a0.Copy(x)
		m.forward(true)

		q := to.Cols

		for j := 0; j < q; j++ {
			d := m.a2.At(0, j) - y.At(0, j)
			ci += d * d
		}
	}

	c := ci / float32(n)

	m.cd = m.c - c
	m.c = c

	return c
}

func main() {
	td := []float32{
		0, 0, 0,
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	}

	stride := 3
	n := len(td) / stride

	ti := NewMat(n, 2, "ti")
	ti.SetStride(stride)
	ti.SetArr(td)

	to := NewMat(n, 1, "to")
	to.SetStride(stride)
	to.SetArr(td[2:])

	var m NN

	m.fp = 0.5
	m.cd = 0

	m.a0 = NewMat(1, 2, "x")

	m.w1 = NewMat(2, 2, "w1")
	m.b1 = NewMat(1, 2, "b1")
	m.nw1 = NewMat(2, 2, "nw1")
	m.nb1 = NewMat(1, 2, "nb1")
	m.a1 = NewMat(1, 2, "a1")

	m.w2 = NewMat(2, 1, "w2")
	m.b2 = NewMat(1, 1, "b2")
	m.nw2 = NewMat(2, 1, "nw2")
	m.nb2 = NewMat(1, 1, "nb2")
	m.a2 = NewMat(1, 1, "a2")

	m.w1.Rand(0, 1)
	m.b1.Rand(0, 1)
	m.nw1.Fill(1)
	m.nb1.Fill(1)
	m.w2.Rand(0, 1)
	m.b2.Rand(0, 1)
	m.nw2.Fill(1)
	m.nb2.Fill(1)

	m.nw1.Flip(m.fp)
	m.nb1.Flip(m.fp)

	m.nw2.Flip(m.fp)
	m.nb2.Flip(m.fp)

	m.a0.Set(0, 0, 0)
	m.a0.Set(0, 1, 1)

	fmt.Printf("init m.cd = %f, m.c = %f\n", m.cd, m.c)
	fmt.Printf("cost = %f\n", m.cost(*ti, *to))
	fmt.Printf("init m.cd = %f, m.c = %f\n", m.cd, m.c)

	for k := 0; k < 20; k++ {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				m.a0.Set(0, 0, float32(i))
				m.a0.Set(0, 1, float32(j))

				m.forward(false)
				// fmt.Printf("m.cd = %f, m.c = %f\n", m.cd, m.c)

				// fmt.Printf("%d ^ %d = %f\n", i, j, m.a2.At(0, 0))
			}
		}

		fmt.Printf("cost = %f\n", m.cost(*ti, *to))
	}
}
