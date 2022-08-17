from cell_relevancy import *
from collections import defaultdict
import heapq
import json
# not used anymore
def mst_prim(graph_matrix):
        visited = set([0])
        my_queue = [(graph_matrix[0][i], (0,i)) for i in range(len(graph_matrix))]
        heapq.heapify(my_queue)
        mst = []
       
        while len(my_queue)>0:
            # print(f"current queue{my_queue}")
            weight, edges = heapq.heappop(my_queue)
            # print(f"test {edges[0]},  {edges[1]}, {visited}")
            if (edges[0] not in visited) or (edges[1] not in visited):
                cur_vertex = egdes[0] if edges[0] not in visited else edges[1]
                # print(f"cur_vertex is {cur_vertex}")
                visited.add(cur_vertex)
                mst.append((weight, edges))
                for i in range(len(graph_matrix)):
                    if i not in visited:
                        heapq.heappush(my_queue, (graph_matrix[cur_vertex][i], (cur_vertex, i)))
        
        return mst


class union_find:
    parent_node = {}
    rank = {}

    def make_set(self, u):
        for i in u:
            self.parent_node[i] = i
            self.rank[i] = 0

    def op_find(self, k):
        if self.parent_node[k] != k:
            self.parent_node[k] = self.op_find(self.parent_node[k])
        return self.parent_node[k]

    def op_union(self, a, b):
        x = self.op_find(a)
        y = self.op_find(b)

        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent_node[y] = x
        elif self.rank[x] < self.rank[y]:
            self.parent_node[x] = y
        else:
            self.parent_node[x] = y
            self.rank[y] = self.rank[y] + 1


#LAYOUT API
def get_layout(all_cells):
    #assume the bullet point, and type producer is annototation_generator(cell)
    #XUYE'S ENDPOINT
    def annotation_generator(cell): # only for testing purposes, substited with real implementations later 
        return "test annotation", "type"

    outputs_dict = defaultdict(lambda:[])
    cell_collectons = []
    for cell in all_cells:
        if cell["cellType"] == "Code":
            bullet, bullet_type =  annotation_generator(cell)
            single_bullet_dict = dict()
            # 
            # bullets{
                # cell_id:
                # bullet_id:
                # bullet:
                # isChosen: True
                # type: code or comment
            # }
            #construct bullet dict
            single_bullet_dict["cell_id"] = cell["cellID"]
            single_bullet_dict["bullet_id"] =  cell["cellID"].replace("c", "b")
            single_bullet_dict["bullet"] = bullet
            single_bullet_dict["isChosen"] = True
            single_bullet_dict["type"]= bullet_type
            outputs_dict["bullets"].append(single_bullet_dict)

            cell_struct  = CellStruct(cell["cellID"], cell["order"], cell["media"], bullet, AST_Code(cell["source"]))
            cell_collectons.append(cell_struct)
    sorted_cell_collectons = sorted(cell_collectons, key=lambda x: int(x.order)) #sort by execution order

    cell_dict = {i:cell_collection for i, cell_collection in enumerate(sorted_cell_collectons)} # key: relative order, value: cell_group
   
    threshold = 0.5 # tune in the future
    data = union_find()
    cell_counts = len(sorted_cell_collectons)
    data.make_set(range(cell_counts))
    for i in range(cell_counts):
        for j in range(i, cell_counts):
            if compute_relevancy(cell_dict[i].ast_code, cell_dict[j].ast_code) > threshold:
                data.op_union(i,j)

    # gather groups
    cell_group_dict = defaultdict(lambda: [])
    for i in range(cell_counts):
        cell_group_dict[data.op_find(i)].append(cell_dict[i])
    

    for _, v in cell_group_dict.items():
        group_output_dict = dict()
        group_output_dict["group_size"] = len(v)
        group_output_dict["bullets"] = []
        group_output_dict["media"] = []
        sorted_v = sorted(v, key=lambda x: int(x.order))
        for _i, _v in enumerate(sorted_v):
            if _v.bullets:
                bullets_dict = dict()
                bullets_dict["cellID"] = _v.cell_id
                bullets_dict["bullets"] = _v.cell_id.replace("c", "b")
                bullets_dict["groupID"] = _i 
                group_output_dict["bullets"].append(bullets_dict)

            if _v.media:
                for _m in _v.media:
                    media_dict = dict()
                    media_dict["mediaID"] = _m["mediaID"]
                    media_dict["isChosen"] = _m["isChosen"]
                    media_dict["groupID"] = _i
                    group_output_dict["media"].append(media_dict)
        outputs_dict["layout"].append(group_output_dict)
    print("layout", outputs_dict)
    return dict(outputs_dict) #convert default dict back to dict



#RELEVENT CELL API
def get_relevent_cells(all_cells):

    rest_code_dict = dict()
    rest_code_weight = defaultdict(lambda x: [])
    current_selected = [cells for cells in all_cells if cells["isChosen"] == True and cells["cellType"] == "Code"]
    if len(current_selected) != 1: # we do not do any cell relevancy prediction if the chosen cell is markdown
        return []

    current_AST =  AST_Code(current_selected[0]["source"])
    # print(f'all code cells are:{all_code_cells}')
    for cell in [cells for cells in all_cells if cells["cellType"] == "Code" and cells["isChosen"] == False]:
        rest_code_dict[cell["cellID"]] = cell["source"]

    weights = []
    for code_id, compared_code in rest_code_dict.items():
        compared_AST = AST_Code(compared_code)
        weights.append((compute_relevancy(current_AST, compared_AST), code_id))
        
    weights.sort(reverse = True)
    
    # compose backend response
    response = []
    for weight_pair in weights[:3]: # set to pick top 3 relevant
        res_dict = dict()
        res_dict["source"] = current_selected[0]["cellID"]
        res_dict["target"] = weight_pair[1]
        res_dict["weight"] = weight_pair[0]
        response.append(res_dict)
    print("response", response)
    return response
        

 
            



    
    
all_code_cells = [{"cellID":"c-0", "cellType": "Code", "isChosen": True,
    "source":"from sklearn.linear_model import LogisticRegression\nlog_clf = LogisticRegression(max_iter = 1000, random_state = 4)\nlog_clf.fit(x_train, y_train)\nlog_score = log_clf.score(x_test, y_test)\nlog_score",
    "outputs": [{"name":"stdout", "text":"FF\nFF\n", "output_type":"stream"}]}, {"cellID":"c-1", "cellType": "Code", "isChosen": False,
    "source":"log_grid = {'C': np.logspace(-4, 4),'solver': ['liblinear'], 'max_iter': np.arange(100, 2000, 100),'penalty':['l1', 'l2']}\nlog_gscv = GridSearchCV(LogisticRegression(max_iter = 1000, random_state = 7),param_grid=log_grid,cv=5,verbose=True)\nlog_gscv.fit(x_train, y_train)\nlog_tuned_score = log_gscv.score(x_test, y_test)\nlog_tuned_score",
    "outputs": [{"name":"stdout", "text":"FF\nFF\n", "output_type":"stream"}]}
    ]
get_relevent_cells(all_code_cells)
# get_layout(all_code_cells)



    

