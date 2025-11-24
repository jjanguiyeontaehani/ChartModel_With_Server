import json
import time
import os
import sys

from django.shortcuts import render

from repositories.models import Stock, StockData
from services.controller import retrieve_stock_data_from_db, retrieve_model_status_from_db, retrieve_stock_list_from_db

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.config.config import ModelConfig
config = ModelConfig()


def index(request):
    trained_models = retrieve_stock_list_from_db()
    
    context = {
        "page_title": "Stock Prediction Dashboard",
        "trained_models": trained_models,
        "chart_data": {},
        "start_date": time.strftime('%Y-%m-%d', time.gmtime(time.time() - config.window_size* 24 * 60 * 60)),
        "end_date": time.strftime('%Y-%m-%d', time.gmtime())
    }


    for model in trained_models:
        context["chart_data"][model] = retrieve_stock_data_from_db(model, context["start_date"], context["end_date"])

    return render(request, 'index.html', context)