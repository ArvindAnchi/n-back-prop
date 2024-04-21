package main

import (
	"log"
	"math"
	"math/rand/v2"
)

func sigmoid(x float32) float32 {
	return 1 / float32(1+math.Exp(float64(-x)))
}

func cost(train [4][3]float32, w1 float32, w2 float32, b float32) float32 {
	var res float32 = 0

	for i := range train {
		x1 := train[i][0]
		x2 := train[i][1]

		y := sigmoid(w1*x1 + w2*x2 + b)

		d := y - train[i][2]

		res += d * d
	}

	return res / float32(len(train))
}

func main() {
	train := [4][3]float32{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 1},
	}

	var eps float32 = 1e-2
	var lr float32 = 1e-1

	w1 := float32(rand.Float32())
	w2 := float32(rand.Float32())

	b := float32(rand.Float32())

	log.Printf("Initial cost: %f W1: %f W2: %f B: %f", cost(train, w1, w2, b), w1, w2, b)

	for i := 0; i < 4000; i++ {
		c := cost(train, w1, w2, b)

		dw1 := (cost(train, w1+eps, w2, b) - c) / eps
		dw2 := (cost(train, w1, w2+eps, b) - c) / eps
		db := (cost(train, w1, w2, b+eps) - c) / eps

		w1 -= lr * dw1
		w2 -= lr * dw2
		b -= lr * db
	}

	log.Printf("Final cost: %f W1: %f W2: %f B: %f", cost(train, w1, w2, b), w1, w2, b)

	log.Print("------------")

	for i := range train {
		x1 := train[i][0]
		x2 := train[i][1]

		y := sigmoid(w1*x1 + w1*x2 + b)

		log.Printf("%f %f %f", x1, x2, y)
	}
}
