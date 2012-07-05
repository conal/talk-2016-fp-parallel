TARG = what-is-functional-programming

# .PRECIOUS: %.tex

default: $(TARG).pdf

theme = Warsaw
# theme = default
theme = Madrid

beamerOpts = -V theme:$(theme)
beamerOpts += --incremental

# Hack to stop pandoc from changing list item indents
tweak = sed -e 's/\\renewcommand{\\@listi}/\\newcommand{\\voot}/'

%.tex: %.md makefile
	pandoc $*.md -t beamer --standalone $(beamerOpts) | $(tweak) > $*.tex

%.pdf: %.tex makefile
	pdflatex $*.tex

%-s5.html: %.md makefile
	pandoc -t s5 -s $*.md -o $*-s5.html --standalone

%-slidy.html: %.md makefile
	pandoc -t slidy -s $*.md -o $*-slidy.html --standalone --incremental

# showpdf=open
# showpdf=evince
# showpdf=explorer
showpdf = open -a Skim.app

see: $(TARG).see

%.see: %.pdf
	${showpdf} $*.pdf


.PRECIOUS: foo.pdf
