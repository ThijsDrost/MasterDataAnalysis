import ast
import os
from typing import Literal
import warnings
import pathlib


def docstring(loc, skip_names=('.idea', '.git', '__init__', '__pycache__'), depth: Literal['file', 'dir'] = 'dir'):
    def check_skip(name):
        return any([(skip in name) for skip in skip_names])

    def extract_docstring(loc, depth) -> (str, dict):
        doc = pathlib.Path(loc).read_text()

        if depth == 'dir':
            doc = doc.split('\n')
            if doc[0].startswith('"""'):
                doc[0] = doc[0].removeprefix('"""')
            else:
                warnings.warn(f"File {loc.removeprefix(r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis')} does not start with a docstring")
                return '', {}
            for index, line in enumerate(doc):
                if '"""' in line:
                    doc[index] = doc[index].removesuffix('"""')
                    doc = doc[:index+1]
                    break
            doc = '\n'.join(doc)
            doc = doc.replace('\n\n', r'\\')
            doc = doc.split('\n')
            return (' '.join((d.strip() for d in doc))).strip().replace('_', '\\_').replace(r'\\', '\n\n'), {}
        elif depth == 'file':
            return docstring_from_file(loc)
        else:
            raise ValueError("`Depth` should be 'dir' or 'file'")

    def _docstring(loc, curr_dict: dict, depth: str):
        file_name = loc.split('\\')[-1].replace('_', '\\_')
        if os.path.isfile(loc):
            curr_dict[file_name] = extract_docstring(loc, depth)
        else:
            curr_dict[file_name] = '', {}
            if '__init__.py' in os.listdir(loc):
                curr_dict[file_name] = extract_docstring(os.path.join(loc, '__init__.py'), depth)
            elif 'docs.txt' in os.listdir(loc):
                curr_dict[file_name] = extract_docstring(os.path.join(loc, 'docs.txt'), depth)

            for item in os.listdir(loc):
                if check_skip(item):
                    continue
                if os.path.isfile(os.path.join(loc, item)) and (not item.endswith('.py')):
                    continue
                _ = _docstring(os.path.join(loc, item), curr_dict[file_name][1], depth)
            return curr_dict

    return _docstring(loc, {}, depth)


def docstring_from_file(loc) -> tuple[str, dict]:
    code = ast.parse(pathlib.Path(loc).read_text())
    doc_dict = {}

    def _node(node, doc_dict):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            doc_dict[node.name] = docstring
        if isinstance(node, ast.ClassDef):
            if node.name.startswith('_') and node.name != '__init__':
                return
            docstring = ast.get_docstring(node)
            doc_dict[node.name] = docstring, {}
            for sub_node in ast.iter_child_nodes(node):
                _node(sub_node, doc_dict[node.name][1])

    if isinstance(code, ast.Module):
        docstring = ast.get_docstring(code)
    else:
        raise ValueError('Loc is not a file')
    for node in ast.iter_child_nodes(code):
        _node(node, doc_dict)
    return docstring, doc_dict


def _write_latex_documentation(file_handle, docstring_dict: dict, depth=0, depth_funcs=None):
    if depth_funcs is None:
        depth_funcs = [LatexHeaders.section, LatexHeaders.subsection, LatexHeaders.subsubsection, LatexHeaders.paragraph,
                       LatexHeaders.subparagraph]
    if depth >= len(depth_funcs):
        raise ValueError(f"`depth_funcs` should be at least as long {len(depth_funcs)} as the depth ({depth})")

    def sort_func(x):
        if x[0].endswith('.py'):
            return '1' + x[0]
        else:
            return '0' + x[0]

    for key, value in sorted(docstring_dict.items(), key=sort_func):
        file_handle.write(depth_funcs[depth](key))
        file_handle.write(value[0] + '\n')
        if value[1] != {}:
            _write_latex_documentation(file_handle, value[1], depth + 1, depth_funcs)


def write_simple_latex_docs(loc, out_loc):
    with open(out_loc, 'w') as file:
        docstring_dict = docstring(loc)
        _write_latex_documentation(file, docstring_dict)


def write_full_latex_docs(loc, out_loc):
    with open(out_loc, 'w') as file:
        docstring_dict = docstring(loc, depth='file')
        _write_latex_documentation(file, docstring_dict)


def find_folder_num(loc, skip_names=('.idea', '.git', '__pycache__')):
    """Recursively find the number of folders in a directory"""
    if not os.path.isdir(loc) or any([skip in loc for skip in skip_names]):
        return 0
    else:
        return 1 + sum([find_folder_num(os.path.join(loc, item)) for item in os.listdir(loc)])


def find_file_num(loc: str, endswith='.py'):
    """Recursively find the number of files in a directory"""
    if os.path.isfile(loc):
        if loc.endswith(endswith):
            return 1
        else:
            return 0
    else:
        return sum([find_file_num(os.path.join(loc, item)) for item in os.listdir(loc)])


def calc_code_lines(loc: str, skip_names=('.idea', '.git', '__pycache__'), endswith='.py'):
    """Recursively find the number of lines of code in a directory"""
    if os.path.isfile(loc):
        if loc.endswith(endswith):
            lines = [line for line in pathlib.Path(loc).read_text().split('\n') if line.strip()]
            if len(lines) == 0:
                if '__init__' in loc:
                    return 0
                print(f"File {loc} has no code")
            return len(lines)
        else:
            return 0
    elif any([skip in loc for skip in skip_names]):
        return 0
    else:
        return sum([calc_code_lines(os.path.join(loc, item)) for item in os.listdir(loc)])


class LatexHeaders:
    @staticmethod
    def chapter(title: str):
        return f"\\chapter{{{title}}}"

    @staticmethod
    def section(title: str):
        return f"\\section{{{title}}}"

    @staticmethod
    def subsection(title: str):
        return f"\\subsection{{{title}}}"

    @staticmethod
    def subsubsection(title: str):
        return f"\\subsubsection{{{title}}}"

    @staticmethod
    def paragraph(title: str):
        return f"\\paragraph{{{title}}}"

    @staticmethod
    def subparagraph(title: str):
        return f"\\subparagraph{{{title}}}"

    @staticmethod
    def italics(text: str):
        return f'\\textit{{{text}}}'


