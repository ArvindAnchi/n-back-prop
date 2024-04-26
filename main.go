package main

import (
	"fmt"

	. "github.com/ArvindAnchi/n_back_prop/matrix"
)

type NN struct {
	cc float32

	a0 *Mat

	w1, b1, a1 *Mat
	w2, b2, a2 *Mat
}

const LR = 2e-1

func (m *NN) forward() {
	m.a1.Dot(m.a0, m.w1)
	m.a1.Sum(m.b1)
	m.a1.Sigmoid()

	m.a2.Dot(m.a1, m.w2)
	m.a2.Sum(m.b2)
	m.a2.Sigmoid()
}

func (m *NN) train(c float32) {
	cd := m.cc - c
	m.cc = c

	m.w1.Nudge(m.w1, cd)
	m.b1.Nudge(m.b1, cd)

	m.w2.Nudge(m.w2, cd)
	m.b2.Nudge(m.b2, cd)
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
		m.forward()

		q := to.Cols

		for j := 0; j < q; j++ {
			d := m.a2.At(0, j) - y.At(0, j)
			ci += d * d
		}
	}

	c := ci / float32(n)

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

	m.a0 = NewMat(1, 2, "x")

	m.w1 = NewMat(2, 2, "w1")
	m.b1 = NewMat(1, 2, "b1")
	m.a1 = NewMat(1, 2, "a1")

	m.w2 = NewMat(2, 1, "w2")
	m.b2 = NewMat(1, 1, "b2")
	m.a2 = NewMat(1, 1, "a2")

	m.a0.Set(0, 0, 0)
	m.a0.Set(0, 1, 1)

	m.w1.Rand(0, 1)
	m.b1.Rand(0, 1)
	m.w2.Rand(0, 1)
	m.b2.Rand(0, 1)

	fmt.Printf("cost = %f\n", m.cost(*ti, *to))

	for k := 0; k < 100; k++ {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				m.a0.Set(0, 0, float32(i))
				m.a0.Set(0, 1, float32(j))

				m.forward()
				c := m.cost(*ti, *to)
				m.train(c)

				fmt.Printf("cost = %f\n", c)

				// fmt.Printf("%d ^ %d = %f\n", i, j, m.a2.At(0, 0))
			}
		}
	}
}
