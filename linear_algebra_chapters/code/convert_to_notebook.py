"""
Jupyter Notebook 转换器
========================

这个脚本将 .py 文件转换为.ipynb格式的 Jupyter Notebook。

使用方式:
    python convert_to_notebook.py
    
会生成对应的.ipynb文件，可以直接用 jupyter notebook 或 jupyter lab 打开。
"""

import json
import os


def py_to_ipynb(py_file, output_dir='converted'):
    """将 Python 脚本转换为 Jupyter Notebook"""
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取文件名
    filename = os.path.basename(py_file)
    base_name = filename.replace('.py', '')
    
    # 解析 Python 文件为单元格
    cells = []
    
    # 分割为代码块和文档块 (基于三引号字符串)
    lines = content.split('\n')
    current_cell_lines = []
    cell_type = 'code'
    
    in_docstring = False
    
    for line in lines:
        if '"""' in line or "'''" in line:
            # 检查是否在 docstring 中
            quote = '"""' if '"""' in line else "'''"
            count = line.count(quote)
            
            if not in_docstring:
                if count >= 2:
                    # 单行 docstring，跳过
                    continue
                else:
                    in_docstring = True
                    cell_type = 'markdown'
                    current_cell_lines = [line]
            else:
                if count >= 2:
                    in_docstring = False
                    cells.append({
                        'cell_type': cell_type,
                        'source': '\n'.join(current_cell_lines),
                        'metadata': {}
                    })
                    current_cell_lines = []
                else:
                    current_cell_lines.append(line)
        elif line.strip().startswith('def ') or line.strip().startswith('if __name__'):
            # 开始新的代码块
            if current_cell_lines and cell_type == 'code':
                cells.append({
                    'cell_type': 'code',
                    'source': '\n'.join(current_cell_lines),
                    'metadata': {},
                    'execution_count': None,
                    'outputs': []
                })
            
            if not in_docstring:
                current_cell_lines = [line]
        elif line.strip() and not in_docstring:
            # 代码行
            current_cell_lines.append(line)
    
    # 添加最后一个单元格
    if current_cell_lines:
        cells.append({
            'cell_type': cell_type,
            'source': '\n'.join(current_cell_lines),
            'metadata': {}
        })
    
    # 创建 Notebook 结构
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'name': 'python',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    
    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入.ipynb文件
    ipynb_file = os.path.join(output_dir, base_name + '.ipynb')
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成：{py_file} → {ipynb_file}")
    return ipynb_file


def main():
    """批量转换所有.py文件"""
    
    import glob
    
    # 查找当前目录下的.py文件 (排除 convert_to_notebook.py 自身)
    py_files = glob.glob('*.py')
    py_files = [f for f in py_files if f != 'convert_to_notebook.py']
    
    if not py_files:
        print("没有找到 .py 文件!")
        return
    
    print(f"找到 {len(py_files)} 个 Python 文件，开始转换...\n")
    
    converted_count = 0
    for py_file in py_files:
        try:
            ipynb_file = py_to_ipynb(py_file)
            converted_count += 1
        except Exception as e:
            print(f"❌ 转换失败：{py_file} - {e}")
    
    print(f"\n{'='*70}")
    print(f"完成！共转换 {converted_count}/{len(py_files)} 个文件")
    print(f"生成的 Notebook 可以在当前目录找到 (.ipynb文件)")


if __name__ == "__main__":
    main()
