CARGO := cargo --offline

.PHONY: all test debug rel release clean

all: debug

test:
	$(CARGO) test --lib

debug:
	$(CARGO) build --lib --bins --examples

rel: release

release:
	$(CARGO) build --release --lib --bins --examples

clean:
	rm -rf target
