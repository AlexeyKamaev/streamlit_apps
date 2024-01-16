import random
import numpy as np
import pandas as pd
import scipy

import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import streamlit as st


tips = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')


from datetime import datetime as DT
from datetime import timedelta

start_dt = DT.strptime('01.01.2023', '%d.%m.%Y')
end_dt = DT.strptime('31.01.2023', '%d.%m.%Y')

def get_random_date(start, end):
    return start + timedelta(random.randint(0, (end - start).days))



tips['time_order'] = [get_random_date(start_dt, end_dt) for _ in range(244)]

day = tips.groupby(pd.Grouper(key='time_order',freq='D')).agg({'tip':'sum'}).reset_index()
x = day.time_order
y = day.tip

fig = go.Figure()
fig.add_trace(go.Scatter(x=day.time_order, y=day.tip, name = 'Apple',
                         mode='lines+markers',
                         marker=dict(color=day.tip)))
fig.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  legend=dict(x=.5, xanchor="center"),
                  title="",
                  hovermode="x",
                  template='seaborn',
                  yaxis_title="Количество",
                  xaxis_title="Размер")

fig2 = go.Figure(data=[go.Histogram(x=tips['total_bill'])])
fig2.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  legend=dict(x=.5, xanchor="center"),
                  title="",
                  hovermode="x",
                  template='seaborn',
                  yaxis_title="Количество",
                  xaxis_title="Размер")

data2_1 = tips['total_bill'].to_list()
fig2_1 = ff.create_distplot([data2_1], ['Данные'], curve_type='normal')
fig2_1.update_layout( margin=dict(l=15, r=20, t=30, b=10),legend=dict(x=.5, xanchor="center"),
    xaxis=dict(title='Значение'),
    template='seaborn',
    yaxis=dict(title='Плотность'))


rel = tips.groupby(pd.Grouper(key='time_order',freq='D')).agg({'total_bill':'sum','tip':'sum'}).reset_index()
fig3 = px.scatter(x=rel.time_order,y=rel.tip,color=rel.total_bill)
fig3.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  legend=dict(x=.5, xanchor="center"),
                  title="",
                  hovermode="x",
                  template='simple_white',
                  yaxis_title="Количество",)

data = tips.groupby('size').agg({'total_bill':'mean','tip':'mean'}).round(2)

fig4 = px.scatter(x=data.index,y=data.total_bill,color=data.tip)
fig4.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  title="",
                  hovermode="x",
                  yaxis_title="Размер счета",
                  xaxis_title="Кол-во гостей")
fig4.update_traces(marker=dict(size=12),
                  selector=dict(mode='markers'))



tips['WD']=tips['time_order'].dt.weekday
def day_maker(x):
    a = list(['0 - Понедельник','1 - Вторник','2 - Среда','3 - Четверг','4 - Пятница','5 - Суббота','6 -Воскресенье'])
    return a[x]
tips['WD'] = [day_maker(i) for i in tips['WD']]
wd = tips.groupby('WD').agg({'total_bill':['sum','mean']}).sort_index()

fig5 = px.bar(x=wd.index,y=wd[('total_bill','sum')],color=wd[('total_bill','mean')])
fig5.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  hovermode="x",
                  yaxis_title="Сумма за день",
                  yaxis_range=[375,1100])


fig6 = px.scatter(x=tips.tip,y=tips.WD,color=tips.sex)
fig6.update_traces(marker=dict(size=5),
                  selector=dict(mode='markers'))
fig6.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  hovermode="x",
                  template='simple_white',
                  yaxis_title="День недели",
                  xaxis_title="Размер чаевых")


fig7 = px.box(x=tips.WD,y=tips.total_bill,color=tips.time)
fig7.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  hovermode="x",
                  yaxis_title="Размер счета",
                  )
















import streamlit as st

st.title('Чаевые в ресторане')
st.write('''#1 Динамика чаевых по времени''')
st.plotly_chart(fig)

st.write('''#2 Счета''')
agree = st.checkbox('Галочка')
if not agree:
    st.plotly_chart(fig2)
else:
    st.plotly_chart(fig2_1)

st.write('''#3 Связь размера счёта и чаевых''')
st.plotly_chart(fig3)

st.write('''#4 Связь размера счёта, чаевых и количества гостей''')
st.plotly_chart(fig4)

st.write('''#5 Связь размера счёта и дня недели''')
st.plotly_chart(fig5)

st.write('''#6 Связь чаевых и дня недели''')
st.plotly_chart(fig6)

st.write('''#7 Связь размера счёта и дня недели''')
st.plotly_chart(fig7)

st.write('''#8 Чаевые на завтрак и ужин''')
option = st.selectbox(
    'Choose time',
    ('Lunch', 'Dinner'))
fig8 = go.Figure(data=[go.Histogram(x=tips[(tips['time'] == option)]['tip'])])

fig8.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  legend=dict(x=.5, xanchor="center"),
                  title="",
                  hovermode="x",
                  template='ggplot2',
                  yaxis_title="Размер",
                  xaxis_title="Количество")
st.plotly_chart(fig8)

st.write('''#9 Связь размера счета и чаевых для курящих/некурящих [мужчины / женщины]''')
sex = st.selectbox(
    'Choose sex',
    ('Male', 'Female'))
male = tips.loc[(tips['sex']=='Male'),:]
female = tips.loc[(tips['sex']=='Female'),:]
sexd = {'Male':male, 'Female':female}



fig9 = px.scatter(x=sexd[sex]['total_bill'],y=sexd[sex]['tip'],color=sexd[sex]['smoker'])
st.plotly_chart(fig9)

st.write('''#10 Связь размера счёта и дня недели''')
options = st.multiselect(
    'Choose options:',
    ['size', 'total_bill', 'tip','smoker','time','WD'])

heat_table = tips.copy()

def mutate_time(x):
    if x == 'Dinner':
        return 1
    return 0
def mutate_smoker(x):
    if x == 'No':
        return 1
    return 0
def mutate_WD(x):
    if int(x[0]) >= 5:
        return 1
    return 0

heat_table['time'] = [mutate_time(i) for i in heat_table['time']]
heat_table['smoker'] = [mutate_smoker(i) for i in heat_table['smoker']]
heat_table['WD'] = [mutate_WD(i) for i in heat_table['WD']]


st.write('''#11 Тепловая карта''')
fig10 = fig = px.imshow(heat_table.loc[:,options].corr(), text_auto=True)


st.plotly_chart(fig10)


st.title('Таблица')
st.write(tips.drop('day',axis=1))