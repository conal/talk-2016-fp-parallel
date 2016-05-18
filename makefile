TARG = fp-parallel

.PRECIOUS: %.tex %.pdf %.web

default: $(TARG).pdf

# For after bug that leaves a broken .aux
oops: unaux default

unaux:
	rm -f $(TARG).aux $(TARG).pdf

see: $(TARG).see

%.pdf: %.tex makefile
	pdflatex $*.tex

%.tex: %.lhs macros.tex mine.fmt makefile
	lhs2TeX -o $*.tex $*.lhs

showpdf = open -a Skim.app

%.see: %.pdf
	${showpdf} $*.pdf

clean:
	rm $(TARG).{tex,pdf,aux,nav,snm,ptb}

web: web-token

STASH=conal@conal.net:/home/conal/web/talks
web: web-token

web-token: $(TARG).pdf
	scp $? $(STASH)/$(TARG)-2016.pdf
	touch $@


%.pdf: %.dot makefile
	dot -Tpdf $< -o $@

# .PRECIOUS: %.pdf

push:
	git push
