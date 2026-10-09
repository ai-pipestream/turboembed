// go-potion under the shared timing protocol: one warm pass, then three runs of
// at least two seconds, the best in texts a second. Batch 1 is Encode on
// one goroutine; a larger batch is EncodeMany per chunk (GOMAXPROCS
// goroutines). Model files come from $GO_POTION_HOME/<MODEL>/, seeded from
// the pinned local copies: New downloads nothing when they are there.
package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	potion "github.com/trengrj/go-potion"
)

func timeIt(b, n int, pass func()) {
	pass()
	best := 0.0
	for r := 0; r < 3; r++ {
		start, done := time.Now(), 0
		for time.Since(start) < 2*time.Second {
			pass()
			done += n
		}
		if v := float64(done) / time.Since(start).Seconds(); v > best {
			best = v
		}
	}
	fmt.Printf("go-potion batch %d: %.0f texts/s\n", b, best)
}

func main() {
	kind, textsPath, batchList := os.Args[1], os.Args[2], os.Args[3]
	raw, err := os.ReadFile(textsPath)
	if err != nil {
		panic(err)
	}
	var texts []string
	if err := json.Unmarshal(raw, &texts); err != nil {
		panic(err)
	}
	start := time.Now()
	m, err := potion.New(context.Background(), potion.Model(kind))
	if err != nil {
		panic(err)
	}
	fmt.Printf("go-potion: loaded in %.1f ms\n", float64(time.Since(start).Microseconds())/1e3)
	// "-" times nothing: the run only writes the vectors.
	for _, s := range strings.Split(batchList, ",") {
		if batchList == "-" {
			break
		}
		b, _ := strconv.Atoi(strings.TrimSpace(s))
		timeIt(b, len(texts), func() {
			if b == 1 {
				for _, t := range texts {
					if _, err := m.Encode(t); err != nil {
						panic(err)
					}
				}
				return
			}
			for i := 0; i < len(texts); i += b {
				if _, err := m.EncodeMany(texts[i:min(i+b, len(texts))]); err != nil {
					panic(err)
				}
			}
		})
	}
	if len(os.Args) > 4 {
		f, err := os.Create(os.Args[4])
		if err != nil {
			panic(err)
		}
		w := bufio.NewWriter(f)
		for _, t := range texts {
			// A text the library refuses is a row of NaN, which the score counts.
			v, err := m.Encode(t)
			if err != nil {
				v = make([]float32, m.Dimensions())
				for i := range v {
					v[i] = float32(math.NaN())
				}
			}
			for _, x := range v {
				binary.Write(w, binary.LittleEndian, math.Float32bits(x))
			}
		}
		w.Flush()
		f.Close()
	}
}
