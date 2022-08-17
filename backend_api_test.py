import pytest
import backend_api

def test_get_relevent_cells():
    all_code_cells = [{"cellID":"c-0", "cellType": "Code", "isChosen": True, 
    "source":"from sklearn.linear_model import LogisticRegression\nlog_clf = LogisticRegression(max_iter = 1000, random_state = 4)\nlog_clf.fit(x_train, y_train)\nlog_score = log_clf.score(x_test, y_test)\nlog_score",
    "outputs": [{"name":"stdout", "text":"FF\nFF\n", "output_type":"stream"}]}, {"cellID":"c-1", "cellType": "Code", "isChosen": False,
    "source":"log_grid = {'C': np.logspace(-4, 4),'solver': ['liblinear'], 'max_iter': np.arange(100, 2000, 100),'penalty':['l1', 'l2']}\nlog_gscv = GridSearchCV(LogisticRegression(max_iter = 1000, random_state = 7),param_grid=log_grid,cv=5,verbose=True)\nlog_gscv.fit(x_train, y_train)\nlog_tuned_score = log_gscv.score(x_test, y_test)\nlog_tuned_score",
    "outputs": [{"name":"stdout", "text":"FF\nFF\n", "output_type":"stream"}]}
    ]
    response = backend_api.get_relevent_cells(all_code_cells)
    assert(response[0]["source"] == "c-0")
    assert(response[0]["target"] == "c-1")

    
def test_get_layout():
    all_code_cells = [{"cellID":"c-0", "order": 0, "cellType": "Code", "isChosen": True, 
    "source":"from sklearn.linear_model import LogisticRegression\nlog_clf = LogisticRegression(max_iter = 1000, random_state = 4)\nlog_clf.fit(x_train, y_train)\nlog_score = log_clf.score(x_test, y_test)\nlog_score", 
    "media":[{"mediaID": 'm-0-0', "type": "table", "isChosen": True}],
    "outputs": [{"name":"stdout", "text":"FF\nFF\n", "output_type":"stream"}]}, {"cellID":"c-1","order": 1, "cellType": "Code", "isChosen": False,
    "source":"log_grid = {'C': np.logspace(-4, 4),'solver': ['liblinear'], 'max_iter': np.arange(100, 2000, 100),'penalty':['l1', 'l2']}\nlog_gscv = GridSearchCV(LogisticRegression(max_iter = 1000, random_state = 7),param_grid=log_grid,cv=5,verbose=True)\nlog_gscv.fit(x_train, y_train)\nlog_tuned_score = log_gscv.score(x_test, y_test)\nlog_tuned_score",
    "media":[{"mediaID": 'm-1-0', "type": "chart", "isChosen": True}],
    "outputs": [{"name":"stdout", "text":"FF\nFF\n", "output_type":"stream"}]}
    ]
    expected_res = {'bullets': [{'cell_id': 'c-0', 'bullet_id': 'b-0', 'bullet': 'test annotation', 'isChosen': True, 'type': 'type'}, {'cell_id': 'c-1', 'bullet_id': 'b-1', 'bullet': 'test annotation', 'isChosen': True, 'type': 'type'}], 'layout': [{'group_size': 2, 'bullets': [{'cellID': 'c-0', 'bullets': 'b-0', 'groupID': 0}, {'cellID': 'c-1', 'bullets': 'b-1', 'groupID': 1}], 'media': [{'mediaID': 'm-0-0', 'isChosen': True, 'groupID': 0}, {'mediaID': 'm-1-0', 'isChosen': True, 'groupID': 1}]}]}
    res = backend_api.get_layout( all_code_cells)
    assert(expected_res == res)
test_get_layout()
test_get_relevent_cells()