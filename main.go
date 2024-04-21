package main

import (
	"log"
	"math/rand/v2"
)

func cost(train [][]float32, w float32, b float32) float32 {
	var res float32 = 0

	for i := range train {
		x := train[i][0]
		y := w*x + b

		d := y - train[i][1]

		res += d * d
	}

	return res / float32(len(train))
}

func main() {
	train := [][]float32{
		{0, 0},
		{1, 1},
		{2, 4},
		{3, 6},
		{4, 8},
	}

	var eps float32 = 1e-3
	var lr float32 = 1e-3

	w := float32(rand.Float32())
	b := float32(rand.Float32())

	log.Printf("Initial cost: %f W: %f B: %f", cost(train, w, b), w, b)

	for i := 0; i < 500; i++ {
		c := cost(train, w, b)

		dw := (cost(train, w+eps, b) - c) / eps
		db := (cost(train, w, b+eps) - c) / eps

		w -= lr * dw
		b -= lr * db
	}

	nc := cost(train, w, b)

	log.Printf("Final cost: %f W: %f B: %f", nc, w, b)

	log.Print("------------")

	for i := range train {
		x := train[i][0]
		y := w*x + b

		log.Printf("%f %f", x, y)
	}
}
