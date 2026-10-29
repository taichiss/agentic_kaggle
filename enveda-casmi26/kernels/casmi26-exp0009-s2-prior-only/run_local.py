import json, pathlib
nb = json.loads((pathlib.Path(__file__).parent / 'notebook.ipynb').read_text())
if __name__ == '__main__':  # spawn した子プロセスで再実行しない
    g = {'__name__': '__main__'}
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            exec(''.join(c['source']), g)
