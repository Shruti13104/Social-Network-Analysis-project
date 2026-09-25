
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="InfluenceAI Analysis", page_icon="🚀", layout="wide")
df=pd.read_csv("influencer_dataset.csv")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif}
.block-container{max-width:1400px;padding-top:2rem}
.hero{padding:32px 38px;border-radius:24px;background:linear-gradient(135deg,#171a2b,#30365b);color:#fff;margin-bottom:25px}
.hero h1{font-size:42px;margin:0}.hero p{color:#d9ddec;font-size:16px}
.metric{border:1px solid #e6e7ef;border-radius:16px;padding:18px;background:#fff}
</style>
""",unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🚀 InfluenceAI — Campaign Analysis</h1>
<p>Analyze influencer profiles using engagement, reach, audience fit, campaign cost and ROI signals.</p></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("Campaign Settings")
    category=st.selectbox("Product category",sorted(df.category.unique()))
    audience=st.selectbox("Target audience",sorted(df.audience.unique()))
    platform=st.selectbox("Preferred platform",sorted(df.platform.unique()))
    budget=st.number_input("Marketing budget (₹)",5000,1000000,50000,5000)
    duration=st.slider("Campaign duration (days)",7,90,30)

x=df.copy()
x["platform_fit"]=(x.platform==platform).astype(int)
x["category_fit"]=(x.category==category).astype(int)
x["audience_fit"]=(x.audience==audience).astype(int)
x["budget_fit"]=(x.estimated_cost<=budget).astype(int)
x["reach_norm"]=x.avg_reach/x.avg_reach.max()
x["suitability"]=(x.engagement_rate/12*30)+(x.reach_norm*20)+(x.roi/6.2*20)+(x.audience_match_score/100*10)+(x.platform_fit*8)+(x.category_fit*7)+(x.audience_fit*3)+(x.budget_fit*2)
x["expected_reach"]=(x.avg_reach*(0.75+min(duration,30)/100)).astype(int)
x=x.sort_values("suitability",ascending=False).reset_index(drop=True)
top=x.iloc[0]

a,b,c,d=st.columns(4)
a.metric("Recommended",top.influencer_name)
b.metric("Suitability",f"{top.suitability:.1f}/100")
c.metric("Expected Reach",f"{top.expected_reach:,}")
d.metric("Estimated Cost",f"₹{top.estimated_cost:,.0f}")

st.subheader("Why this influencer?")
reasons=[]
if top.platform_fit: reasons.append(f"Matches {platform}.")
if top.category_fit: reasons.append(f"Works in {category}.")
if top.audience_fit: reasons.append(f"Matches {audience}.")
if top.budget_fit: reasons.append("Estimated cost is within budget.")
reasons.append(f"Engagement rate: {top.engagement_rate:.2f}%.")
reasons.append(f"Historical ROI signal: {top.roi:.2f}x.")
for r in reasons: st.write("✓ "+r)

st.subheader("Top 5 Influencers")
show=x.head(5)[["influencer_name","platform","category","audience","followers","avg_reach","engagement_rate","estimated_cost","roi","suitability"]].copy()
show.columns=["Influencer","Platform","Category","Audience","Followers","Reach","Engagement %","Cost","ROI","Suitability"]
st.dataframe(show.style.format({"Followers":"{:,.0f}","Reach":"{:,.0f}","Engagement %":"{:.2f}%","Cost":"₹{:,.0f}","ROI":"{:.2f}x","Suitability":"{:.1f}"}),use_container_width=True,hide_index=True)

c1,c2=st.columns(2)
with c1:
    fig=px.bar(x.head(5).sort_values("suitability"),x="suitability",y="influencer_name",orientation="h",title="Top 5 Suitability")
    st.plotly_chart(fig,use_container_width=True)
with c2:
    fig=px.scatter(x,x="followers",y="engagement_rate",size="avg_reach",color="category",hover_name="influencer_name",log_x=True,title="Engagement vs Followers")
    st.plotly_chart(fig,use_container_width=True)

c3,c4=st.columns(2)
with c3:
    fig=px.scatter(x,x="avg_reach",y="roi",size="followers",color="platform",hover_name="influencer_name",title="Reach vs ROI")
    st.plotly_chart(fig,use_container_width=True)
with c4:
    cat=x.groupby("category",as_index=False).roi.mean()
    fig=px.bar(cat,x="category",y="roi",title="Average ROI by Category",text_auto=".2f")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Dataset Explorer")
st.dataframe(df,use_container_width=True,hide_index=True)
