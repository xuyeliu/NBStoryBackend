import ast
test_code_A = "from sklearn.linear_model import LogisticRegression\nlog_clf = LogisticRegression(max_iter = 1000, random_state = 4)\nlog_clf.fit(x_train, y_train)\nlog_score = log_clf.score(x_test, y_test)\nlog_score"
test_code_B = "log_grid = {'C': np.logspace(-4, 4),'solver': ['liblinear'], 'max_iter': np.arange(100, 2000, 100),'penalty':['l1', 'l2']}\nlog_gscv = GridSearchCV(LogisticRegression(max_iter = 1000, random_state = 7),param_grid=log_grid,cv=5,verbose=True)\nlog_gscv.fit(x_train, y_train)\nlog_tuned_score = log_gscv.score(x_test, y_test)\nlog_tuned_score"

try:
    _basestring = basestring
except NameError:
    _basestring = str

class AST_Code:
    def __init__(self, code):
        self.code = code
        self.tree = ast.parse(code)
        self.variable = set()
        self.fun_call = set()
                        
        nodes = _strip_docstring(self.tree.body)

        for node in nodes:
            recurse_through_ast(node, self.variable, self.fun_call)
    
def compute_relevancy(src:AST_Code, dest:AST_Code):
        # print(src.variable)
        # print(dest.variable)
        intersection = src.variable.intersection(dest.variable)
        # src_dest_weight = len(intersection)/len(src.variable)
        src_dest_weight = 1 - (1-len(intersection)/len(src.variable))*(1-len(intersection)/len(dest.variable))
        # print(27)
        # print(src_dest_weight)
        return src_dest_weight

        
class CellStruct:
    def __init__(self, cell_id, order, media, bullets, ast_code):
        self.cell_id = cell_id 
        self.order = order
        self.media = media # output = code execution output if cell type == code, else None
        self.bullets = bullets # output == generated annotation if cell type == code, else markdown source
        self.ast_code = ast_code # output == code ast if cell type == code, else None
        self.neighbors = [] #directed acyclic graph

def _strip_docstring(body):
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Str):
        return body[1:]
    return body


def recurse_through_ast(node, variable, fun_call):
    possible_docstring = isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
    node_fields = zip(
        node._fields,
        (getattr(node, attr) for attr in node._fields)
    )
    for field_name, field_value in node_fields:
        if isinstance(field_value, ast.AST):
            #print(f'ast case { field_name}{field_value}')
            recurse_through_ast(field_value, variable, fun_call)
        elif isinstance(field_value, list):
            #print(f'list case { field_name}{field_value}')
            if possible_docstring and omit_docstrings and field_name == 'body':
                field_value = _strip_docstring(field_value)
            for item in field_value:
                    recurse_through_ast(item, variable, fun_call)
        else:
            if field_name == "id":
                variable.add(field_value)
            elif field_name == "attr":
                fun_call.add(field_value)
            #print(f'48 field name is {field_name}. field value is {field_value}')



    



compute_relevancy(AST_Code(test_code_A), AST_Code(test_code_B))


    
