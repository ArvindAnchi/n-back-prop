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

type GEN struct {
	ti, to *Mat
	io     *Mat
}

const LR = 2e-1

func (m *NN) cost(gen GEN) float32 {
	if gen.ti.Rows != gen.to.Rows {
		panic(fmt.Sprintf("M_COST: [%s] Expected shape:(%d %d) got (%d %d)", gen.ti.Name, gen.to.Rows, m.a2.Cols, gen.ti.Rows, gen.ti.Cols))
	}
	if gen.to.Cols != m.a2.Cols {
		panic(fmt.Sprintf("M_COST: [%s] Expected shape:(%d %d) got (%d %d)", gen.ti.Name, gen.to.Rows, m.a2.Cols, gen.to.Rows, gen.to.Cols))
	}

	n := gen.ti.Rows
	var ci float32 = 0

	for i := 0; i < n; i++ {
		x := gen.ti.Row(i)
		y := gen.to.Row(i)

		m.a0.Copy(x)
		m.forward(nil)

		q := gen.to.Cols

		for j := 0; j < q; j++ {
			d := m.a2.At(0, j) - y.At(0, j)
			ci += d * d
		}
	}

	c := ci / float32(n)

	return c
}

func (m *NN) forward(gen *GEN) {
	m.a1.Dot(m.a0, m.w1)
	m.a1.Sum(m.b1)
	m.a1.Sigmoid()

	m.a2.Dot(m.a1, m.w2)
	m.a2.Sum(m.b2)
	m.a2.Sigmoid()

	if gen != nil {

	}
}

func main() {
	td := []float32{
		0, 0, 1, 0,
		0, 1, 0, 1,
		1, 0, 0, 1,
		1, 1, 1, 0,
	}

	stride := 4
	n := len(td) / stride

	var m NN
	var g GEN

	g.ti = NewMat(n, 2, "ti")
	g.to = NewMat(n, 2, "to")

	g.ti.SetStride(stride)
	g.ti.SetArr(td)

	g.to.SetStride(stride)
	g.to.SetArr(td[2:])

	m.a0 = NewMat(1, 2, "a0")

	m.w1 = NewMat(2, 2, "w1")
	m.b1 = NewMat(1, 2, "b1")
	m.a1 = NewMat(1, 2, "a1")

	m.w2 = NewMat(2, 2, "w2")
	m.b2 = NewMat(1, 2, "b2")
	m.a2 = NewMat(1, 2, "a2")

	m.a0.Set(0, 0, 0)
	m.a0.Set(0, 1, 1)

	m.w1.Rand(0, 1)
	m.b1.Rand(0, 1)
	m.w2.Rand(0, 1)
	m.b2.Rand(0, 1)

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			m.a0.Set(0, 0, float32(i))
			m.a0.Set(0, 1, float32(j))

			m.forward(nil)

			fmt.Printf("%d ^ %d = %f\n", i, j, m.a2.At(0, 1)-m.a2.At(0, 0))
		}
	}

	fmt.Print("------\n")

	for k := 0; k < 10; k++ {
		fmt.Print("[T] Cost: ")

		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				m.a0.Set(0, 0, float32(i))
				m.a0.Set(0, 1, float32(j))

				m.forward(&g)
				c := m.cost(g)

				fmt.Printf("[%d ^ %d]=%f ", i, j, c)
			}
		}

		fmt.Print("\n")
	}

	fmt.Print("------\n")

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			m.a0.Set(0, 0, float32(i))
			m.a0.Set(0, 1, float32(j))

			m.forward(nil)

			fmt.Printf("%d ^ %d = %f\n", i, j, m.a2.At(0, 1)-m.a2.At(0, 0))
		}
	}
}
