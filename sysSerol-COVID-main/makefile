flist = 1 2 3 4 5 6 8 9 10 11 13 14 B1 B2


all: $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigure.py
	mkdir -p output
	. venv/bin/activate && ./genFigure.py $*

clean:
	rm -rf output venv
 