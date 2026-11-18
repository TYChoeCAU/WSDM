"""Regenerate manuscript Tables 1--4 from released scalar evidence; no model loading."""
import argparse,json
from pathlib import Path
from paper_data import build,render
HEADERS={1:r' & S1 & S2 & S3 \\',2:r'Backbone & \multicolumn{5}{c}{Yelp2018} & \multicolumn{5}{c}{Amazon-Books} & \multicolumn{5}{c}{ML-1M} \\',3:r'Study & Dataset & Backbone & U $\Delta$ & W $\Delta$ & W 95\% CI & F Base & F $\Delta$ & F 95\% CI & DF $\Delta$ \\',4:r'Dataset & Backbone & Protocol & Base & Pop & NPGC & PRM+PL & Res($p$) & CR & Oracle \\'}
COLS={1:'llll',2:'l'+'r'*15,3:'lll'+'r'*7,4:'lll'+'r'*7}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=Path('results'));ap.add_argument('--out',type=Path,default=Path('outputs/tables'));args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 tables=build(args.results)
 for n,rows in tables.items():
  lines=[r'% Manuscript Table '+str(n)+'. Generated from saved CSV evidence.',r'% Requires booktabs and the manuscript macro \newcommand{\Score}[2]{\ensuremath{#1_{\pm #2}}}.',r'\begin{tabular}{'+COLS[n]+'}',r'\toprule',HEADERS[n],r'\midrule']
  if n==2:
   lines.insert(-1,'Backbone & '+ ' & '.join(['Base','NPGC','Pop','CR','Orc.']*3)+r' \\')
  for i,r in enumerate(rows):
   lines.append(r['label']+' & '+' & '.join(render(c) for c in r['cells'])+r' \\')
  lines += [r'\bottomrule',r'\end{tabular}']
  (args.out/f'table{n}.tex').write_text('\n'.join(lines)+'\n')
 (args.out/'table_values.json').write_text(json.dumps(tables,indent=2)+'\n')
 print(f'Wrote Tables 1--4 to {args.out}.')
if __name__=='__main__':main()
