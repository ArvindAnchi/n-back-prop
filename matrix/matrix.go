package matrix

import (
	"fmt"
	"math"
	"math/rand/v2"
)

type Mat struct {
	name string
	rows int
	cols int
	es   []float32
}

func sigmoid(x float32) float32 {
	return 1 / float32(1+math.Exp(float64(-x)))
}

func NewMat(rows, cols int, name string) *Mat {
	es := make([]float32, rows*cols)

	return &Mat{
		name: name,
		rows: rows,
		cols: cols,
		es:   es,
	}
}

func (m *Mat) Set(i, j int, val float32) {
	m.es[i*m.cols+j] = val
}

func (m *Mat) At(i, j int) float32 {
	return m.es[i*m.cols+j]
}

func (m *Mat) IdxOf(i, j int) int {
	return i*m.cols + j
}

func (m *Mat) Print() {
	fmt.Printf("%s = [\n", m.name)

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			fmt.Printf("\t%f ", m.At(i, j))
		}

		fmt.Printf("\n")
	}

	fmt.Print("]\n")
}

func (m *Mat) Sigmoid() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.es[m.IdxOf(i, j)] = sigmoid(m.At(i, j))
		}
	}
}

func (m *Mat) Fill(val float32) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.es[m.IdxOf(i, j)] = val
		}
	}
}

func (m *Mat) Rand(low, high float32) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.es[m.IdxOf(i, j)] = rand.Float32()*(high-low) + low
		}
	}
}

func (m *Mat) Dot(a, b *Mat) {
	if a.cols != b.rows {
		panic(fmt.Sprintf("MAT_DOT: Expected a:cols:%d == b:rows:%d", a.cols, b.rows))
	}
	if m.cols != b.cols {
		panic(fmt.Sprintf("MAT_DOT: Expected dest:shape:(%d %d) got (%d %d)", a.rows, b.cols, m.rows, m.cols))
	}
	if m.rows != a.rows {
		panic(fmt.Sprintf("MAT_DOT: Expected dest:shape:(%d %d) got (%d %d)", a.rows, b.cols, m.rows, m.cols))
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.es[m.IdxOf(i, j)] = 0

			for k := 0; k < a.cols; k++ {
				m.es[m.IdxOf(i, j)] += a.At(i, k) * b.At(k, j)
			}
		}
	}
}

func (m *Mat) Row(rowIdx int) *Mat {
	sIdx := m.IdxOf(rowIdx, 0)
	eIdx := sIdx + m.cols

	return &Mat{
		name: fmt.Sprintf("%s[%d:%d]", m.name, sIdx, eIdx),
		rows: 1,
		cols: m.cols,
		es:   m.es[sIdx:eIdx],
	}
}

func (m *Mat) Copy(src *Mat) {
	if m.rows != src.rows {
		panic(fmt.Sprintf("MAT_COPY: Expected shape:(%d %d) got (%d %d)", m.rows, m.cols, src.rows, src.cols))
	}
	if m.cols != src.cols {
		panic(fmt.Sprintf("MAT_COPY: Expected shape:(%d %d) got (%d %d)", m.rows, m.cols, src.rows, src.cols))
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			src.es[src.IdxOf(i, j)] = m.es[m.IdxOf(i, j)]
		}
	}
}

func (m *Mat) Sum(b *Mat) {
	if m.rows != b.rows {
		panic(fmt.Sprintf("MAT_SUM: Expected dest:shape:(%d %d) got (%d %d)", m.rows, m.cols, b.rows, b.cols))
	}

	if m.cols != b.cols {
		panic(fmt.Sprintf("MAT_SUM: Expected dest:shape:(%d %d) got (%d %d)", m.rows, m.cols, b.rows, b.cols))
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.es[m.IdxOf(i, j)] += b.es[b.IdxOf(i, j)]
		}
	}
}
