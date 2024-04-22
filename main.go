package main

import (
	"fmt"

	. "github.com/ArvindAnchi/n_back_prop/matrix"
)

type XORModel struct {
	a0 *Mat

	w1, b1, a1 *Mat
	w2, b2, a2 *Mat
}

func (m *XORModel) forward() {
	m.a1.Dot(m.a0, m.w1)
	m.a1.Sum(m.b1)
	m.a1.Sigmoid()

	m.a2.Dot(m.a1, m.w2)
	m.a2.Sum(m.b2)
	m.a2.Sigmoid()
}

func cost() {
}

func main() {
	var m XORModel

	m.a0 = NewMat(1, 2, "x")

	m.w1 = NewMat(2, 2, "w1")
	m.b1 = NewMat(1, 2, "b1")
	m.a1 = NewMat(1, 2, "a1")

	m.w2 = NewMat(2, 1, "w2")
	m.b2 = NewMat(1, 1, "b2")
	m.a2 = NewMat(1, 1, "a2")

	m.w1.Rand(0, 1)
	m.b1.Rand(0, 1)
	m.w2.Rand(0, 1)
	m.b2.Rand(0, 1)

	m.a0.Set(0, 0, 0)
	m.a0.Set(0, 1, 1)

	m.w1.Print()
	m.w1.Row(0).Print()
	m.w1.Row(1).Print()

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			m.a0.Set(0, 0, float32(i))
			m.a0.Set(0, 1, float32(j))

			m.forward()

			fmt.Printf("%d ^ %d = %f\n", i, j, m.a2.At(0, 0))
		}
	}
}
