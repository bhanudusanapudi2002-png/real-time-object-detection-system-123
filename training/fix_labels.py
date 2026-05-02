from pathlib import Path
p=Path(r'c:\Users\vamsi\OneDrive\Pictures\Desktop\SIMS_PROJECTS\ObjectDetection\data\labeled\labels')
count=0
for f in p.rglob('*.txt'):
    s=f.read_text()
    orig=s
    s=s.replace('\n80 ','\n0 ').replace('\n81 ','\n1 ').replace('\n82 ','\n2 ')
    if s.startswith('80 '):
        s='0 '+s[len('80 '):]
    if s.startswith('81 '):
        s='1 '+s[len('81 '):]
    if s.startswith('82 '):
        s='2 '+s[len('82 '):]
    if s!=orig:
        f.write_text(s)
        count+=1
print('Patched',count,'label files')
