"""
The file used to create the appendix with documentation your currently reading. It extracts all the docstrings from the files in
the project and puts them in a latex files. These files are combined to create the appendix.
"""

from General.latex.Docs import write_simple_latex_docs, find_folder_num, find_file_num, calc_code_lines


base_loc_out = r"E:\OneDrive - TU Eindhoven\Master thesis\Tex\Appendices"
base_loc_in = r'C:\Users\20222772\PycharmProjects\MasterDataAnalysis'
write_simple_latex_docs(rf"{base_loc_in}\Data_analysis", fr'{base_loc_out}\Code\code_doc_Data_analysis.tex')
write_simple_latex_docs(fr"{base_loc_in}\General", fr'{base_loc_out}\Code\code_doc_General.tex')
write_simple_latex_docs(fr"{base_loc_in}\Guis", fr'{base_loc_out}\Code\code_doc_Guis.tex')
write_simple_latex_docs(fr"{base_loc_in}\LatexReport", fr'{base_loc_out}\Code\code_doc_Latex.tex')
folders = find_folder_num(f"{base_loc_in}")
files = find_file_num(f"{base_loc_in}")
lines = calc_code_lines(f"{base_loc_in}")

output = rf""" During this thesis an extensive python project ({folders} folders with {files} python files and a total of {lines}
lines of code\footnote{{This also included all the docstrings}}) was written for data generation and analysis. The full project 
can be found on \href{{https://github.com/ThijsDrost/MasterDataAnalysis}}{{GitHub}}. In this appendix the general structure and 
uses for this project are explained. This is extracted from the docstrings in the project. More specific documentation of the 
python functions is (sometimes) included in the code.

The project consists of three main folders: Data\_analysis, General, GUIs, and Latex. The Data\_analysis folder contains all the\
code for the data analysis, since it has no useful documentation, it is not included in this appendix.

%\input{{Appendices/Code/code_doc_Data_analysis.tex}}
%
\input{{Appendices/Code/code_doc_General.tex}}

\input{{Appendices/Code/code_doc_Guis.tex}}

\input{{Appendices/Code/code_doc_Latex.tex}}"""

with open(rf"{base_loc_out}\Code.tex", 'w', encoding='utf-8') as file:
    file.write(output)

