TARG = output/fp

# TARG = output/what-is-functional-programming

# .PRECIOUS: %.tex

default: $(TARG).pdf

bh = beamer-header.tex

# beamerOpts  = -V theme:$(theme) -V colortheme:$(colortheme) -V outertheme:$(outertheme)
beamerOpts += --include-in-header=$(bh)

# beamerOpts += -V theme:Madrid
# beamerOpts += -V theme:Warsaw
beamerOpts += -V theme:Frankfurt
# beamerOpts += -V theme:Berkeley
# beamerOpts += -V colortheme:albatross

# beamerOpts += --incremental
# beamerOpts += -V toc:true

# beamerOpts += --highlight-style=kate

# Hack to stop pandoc from changing list item indents
tweak = sed -e 's/\\renewcommand{\\@listi}/\\newcommand{\\voot}/'

output/%.tex: %.md makefile $(bh)
	pandoc $*.md -f Markdown+LHS -t beamer $(beamerOpts) | $(tweak) > output/$*.tex

output/%.pdf: output/%.tex makefile
	cd output ; pdflatex $*.tex

# %-s5.html: %.md makefile
# 	pandoc -t s5 -s $*.md -o $*-s5.html --standalone

# %-slidy.html: %.md makefile
# 	pandoc -t slidy -s $*.md -o $*-slidy.html --standalone --incremental

# showpdf=open
# showpdf=evince
# showpdf=explorer
showpdf = open -a Skim.app

see: $(TARG).see

%.see: %.pdf
	${showpdf} $*.pdf


.PRECIOUS: output/fp.pdf output/fp.tex

