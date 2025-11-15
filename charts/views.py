from django.shortcuts import render

def comparison_view(request):
    return render(request, 'charts/comparison.html')