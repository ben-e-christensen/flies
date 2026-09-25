import csv, os, shutil
P = r'E:\Ben Christensen\FLIES\frame_check\scores.csv'
REF = r'E:\Ben Christensen\FLIES\session_2026-09-21_17-15-35\flir_a_cont\p001_000800.png'
ref = os.path.normcase(os.path.abspath(REF))
rows = list(csv.reader(open(P, newline='')))
head, body = rows[0], rows[1:]
keep = [r for r in body if os.path.normcase(os.path.abspath(r[0])) > ref and os.path.exists(r[0])]
print(f'{len(body):,} rows -> {len(keep):,} kept ({len(body)-len(keep):,} dropped)')
shutil.copy2(P, P + '.bak')
w = csv.writer(open(P, 'w', newline=''))
w.writerow(head)
w.writerows(keep)
print('pruned; backup at', P + '.bak')