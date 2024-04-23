package matrix

import (
	"fmt"
	"math"
	"math/rand/v2"
)

type Mat struct {
	name   string
	Rows   int
	Cols   int
	stride int
	es     []float32
}

func sigmoid(x float32) float32 {
	return 1 / float32(1+math.Exp(float64(-x)))
}

func NewMat(rows, cols int, name string) *Mat {
	es := make([]float32, rows*cols)

	return &Mat{
		name:   name,
		Rows:   rows,
		Cols:   cols,
		stride: cols,
		es:     es,
	}
}

func (m *Mat) SetArr(es []float32) {
	m.es = es
}

func (m *Mat) SetStride(stride int) {
	m.stride = stride
}

func (m *Mat) Set(i, j int, val float32) {
	m.es[i*m.stride+j] = val
}

func (m *Mat) At(i, j int) float32 {
	return m.es[i*m.stride+j]
}

func (m *Mat) IdxOf(i, j int) int {
	return i*m.stride + j
}

func (m *Mat) Print() {
	fmt.Printf("%s = [\n", m.name)

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			fmt.Printf("\t%f ", m.At(i, j))
		}

		fmt.Printf("\n")
	}

	fmt.Print("]\n")
}

func (m *Mat) Sigmoid() {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.es[m.IdxOf(i, j)] = sigmoid(m.At(i, j))
		}
	}
}

func (m *Mat) Fill(val float32) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.es[m.IdxOf(i, j)] = val
		}
	}
}

func (m *Mat) Flip(fp float32) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if rand.Float32() < fp {
				m.es[m.IdxOf(i, j)] = -m.es[m.IdxOf(i, j)]
			}
		}
	}
}

func (m *Mat) Nudge(nm *Mat, lr float32) {
	if m.Cols != nm.Cols {
		panic(fmt.Sprintf("MAT_NUDGE: Expected m:shape(%d %d) == nm:shape(%d %d)", m.Rows, m.Cols, nm.Rows, nm.Cols))
	}
	if m.Rows != nm.Rows {
		panic(fmt.Sprintf("MAT_NUDGE: Expected m:shape(%d %d) == nm:shape(%d %d)", m.Rows, m.Cols, nm.Rows, nm.Cols))
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.es[m.IdxOf(i, j)] += nm.es[nm.IdxOf(i, j)] * lr
		}
	}
}

func (m *Mat) Rand(low, high float32) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.es[m.IdxOf(i, j)] = rand.Float32()*(high-low) + low
		}
	}
}

func (m *Mat) Dot(a, b *Mat) {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf("MAT_DOT: Expected a:cols:%d == b:rows:%d", a.Cols, b.Rows))
	}
	if m.Cols != b.Cols {
		panic(fmt.Sprintf("MAT_DOT: Expected dest:shape:(%d %d) got (%d %d)", a.Rows, b.Cols, m.Rows, m.Cols))
	}
	if m.Rows != a.Rows {
		panic(fmt.Sprintf("MAT_DOT: Expected dest:shape:(%d %d) got (%d %d)", a.Rows, b.Cols, m.Rows, m.Cols))
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.es[m.IdxOf(i, j)] = 0

			for k := 0; k < a.Cols; k++ {
				m.es[m.IdxOf(i, j)] += a.At(i, k) * b.At(k, j)
			}
		}
	}
}

func (m *Mat) Row(rowIdx int) *Mat {
	sIdx := m.IdxOf(rowIdx, 0)
	eIdx := sIdx + m.Cols

	return &Mat{
		name:   fmt.Sprintf("%s[%d:%d]", m.name, sIdx, eIdx),
		Rows:   1,
		Cols:   m.Cols,
		stride: m.stride,
		es:     m.es[sIdx:eIdx],
	}
}

func (m *Mat) Copy(src *Mat) {
	if m.Rows != src.Rows {
		panic(fmt.Sprintf("MAT_COPY: Expected shape:(%d %d) got (%d %d)", m.Rows, m.Cols, src.Rows, src.Cols))
	}
	if m.Cols != src.Cols {
		panic(fmt.Sprintf("MAT_COPY: Expected shape:(%d %d) got (%d %d)", m.Rows, m.Cols, src.Rows, src.Cols))
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			src.es[src.IdxOf(i, j)] = m.es[m.IdxOf(i, j)]
		}
	}
}

func (m *Mat) Sum(b *Mat) {
	if m.Rows != b.Rows {
		panic(fmt.Sprintf("MAT_SUM: Expected dest:shape:(%d %d) got (%d %d)", m.Rows, m.Cols, b.Rows, b.Cols))
	}

	if m.Cols != b.Cols {
		panic(fmt.Sprintf("MAT_SUM: Expected dest:shape:(%d %d) got (%d %d)", m.Rows, m.Cols, b.Rows, b.Cols))
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.es[m.IdxOf(i, j)] += b.es[b.IdxOf(i, j)]
		}
	}
}
