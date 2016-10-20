### Save plots to a pdf file
~~~~{.python}
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('output.pdf')
plt.savefig(pp, format='pdf')
pp.savefig()
pp.close()
~~~~
