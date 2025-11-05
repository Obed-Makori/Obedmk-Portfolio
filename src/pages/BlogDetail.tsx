import { useParams, useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import BlogContent from '@/components/BlogContent';
import CommentsSection from '@/components/CommentsSection';

// Import blog images
import mlBiImage from '@/assets/blog-ml-bi.jpg';
import pythonImage from '@/assets/blog-python.jpg';
import powerBiImage from '@/assets/blog-powerbi.jpg';
import etlImage from '@/assets/blog-etl.jpg';
import futureImage from '@/assets/blog-future.jpg';
import sqlImage from '@/assets/blog-sql.jpg';
import dataPipelineImage from '@/assets/blog-data-pipeline.jpg';
import dashboardExampleImage from '@/assets/blog-dashboard-example.jpg';

const BlogDetail = () => {
  const { slug } = useParams();
  const navigate = useNavigate();

  const blogs = [
    {
      id: 1,
      slug: "machine-learning-business-intelligence-guide",
      title: "Machine Learning in Business Intelligence: A Practical Guide",
      excerpt: "Explore how ML algorithms can enhance traditional BI practices and drive better business decisions.",
      content: `
        <h2 style="font-size: 24px;">Introduction</h2>
        <p>In today's data-driven world, businesses are collecting more information than ever before. But raw data alone does not drive growth! It is what you do with it that matters. This is where Machine Learning (ML) steps in. When combined with traditional Business Intelligence (BI), ML transforms static reports into dynamic, predictive insights that help organizations stay ahead of the curve.</p>
        <br>
        <p> Imagine a retail company that has been tracking its sales for years. The Business Intelligence (BI) dashboards show what happened last quarter, which products sold the most, and which stores underperformed. That's useful. But what if the company could go further? What if it could predict next month’s bestsellers, identify customers likely to churn, or even spot fraud before it happens?</p>
        <br>
        <p> This is precisely what happens when Machine Learning (ML) is integrated with BI. Instead of just reporting on the past, ML empowers businesses to forecast the future and recommend actions that can drive growth.</p>
        <br>
        <h2 style="font-size: 24px;">Practical Implementation Strategy</h2>

        <p>Adopting ML in BI is not about jumping straight into complex models. Think of it like teaching a new employee. You start small and build confidence over time.</p>
        <br>
        <p>First, ensure that your data is clean and reliable. No matter how advanced the model is, poor data quality will lead to poor results. Next, establish a solid data governance framework: clear rules for how data is collected, stored, and used. Then, begin with simple models such as basic regression or clustering before moving on to deep learning or neural networks. Finally, invest in people. A BI team equipped with ML skills, or access to data science expertise, becomes a powerful driver of innovation. </p>
        <br>
                  
        <h2 style="font-size: 24px;">Real-World Applications</h2>
        <p>The power of ML in BI is already visible across industries. <a href="https://www.netflix.com" target="_blank" style="color: #0096FF;">Netflix</a> is a great example. While BI reports could tell Netflix which shows performed best last month, ML goes further by predicting what each viewer is likely to watch next. This keeps users engaged and reduces churn.</p>
        <br>
        <p> In the financial sector, fraud detection is another strong use case. Traditional BI tools can report on the number of fraud cases identified in the past, but ML models can detect suspicious behavior in real time. This proactive approach saves institutions millions.</p>
        <br>
        <p>Retailers like Amazon also rely heavily on ML-driven BI. Instead of simply knowing last year’s sales, ML enables Amazon to personalize recommendations for each customer—turning insights into real-time revenue opportunities. </p>
        <br>
        <h2 style="font-size: 20px;">Conclusion</h2>
        <p>Business Intelligence helps companies understand what happened. Machine Learning, on the other hand, helps them predict and act on what will happen next. When combined, they form a powerful toolkit that drives smarter decisions, deeper customer understanding, and stronger competitive advantages.</p>
      `,
      date: "Dec 15, 2024",
      readTime: "8 min read",
      category: "Machine Learning",
      tags: ["ML", "BI", "Data Science"],
      image: mlBiImage
    },
    {
      id: 2,
      slug: "python-data-analysis-essential-libraries",
      title: "Python for Data Analysis: Essential Libraries and Techniques",
      excerpt: "A comprehensive overview of Python libraries that every data analyst should master.",
      content: `
        <h2>Essential Python Libraries</h2>
        <p><strong>Pandas:</strong> The cornerstone of data manipulation in Python. Provides powerful data structures like DataFrames and Series for handling structured data efficiently.</p>
        
        <pre><code class="language-python">import pandas as pd
import numpy as np

# Create a DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Basic operations
print(df.head())
print(df.describe())</code></pre>
        
        <p><strong>NumPy:</strong> Foundation for numerical computing. Offers support for large multi-dimensional arrays and mathematical functions.</p>
        
        <pre><code class="language-python"># NumPy array operations
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")
print(f"Standard deviation: {np.std(arr)}")</code></pre>
        
        <p><strong>Matplotlib & Seaborn:</strong> Visualization libraries for creating static, interactive, and publication-quality plots.</p>
        
        <h2>Data Cleaning Techniques</h2>
        <p>Handle missing values with fillna(), dropna(), and interpolation methods. Use string methods for text cleaning and regex for pattern matching.</p>
        
        <pre><code class="language-python"># Handle missing values
df_cleaned = df.dropna()  # Remove rows with NaN
df_filled = df.fillna(df.mean())  # Fill with mean

# String cleaning
df['name'] = df['name'].str.lower().str.strip()</code></pre>
        
        <h2>Advanced Analysis</h2>
        <p>Leverage groupby operations, pivot tables, and merge/join operations for complex data transformations. Time series analysis with datetime indexing.</p>
        
        <pre><code class="language-python"># GroupBy operations
grouped = df.groupby('department')['salary'].mean()

# Pivot tables
pivot = df.pivot_table(values='salary', index='department', columns='year')

# Time series
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
monthly_avg = df.resample('M').mean()</code></pre>
      `,
      date: "Dec 10, 2024", 
      readTime: "12 min read",
      category: "Data Analysis",
      tags: ["Python", "Pandas", "NumPy"],
      image: pythonImage
    },
    {
      id: 3,
      slug: "building-effective-dashboards-power-bi",
      title: "Building Effective Dashboards: Design Principles for Power BI",
      excerpt: "Learn the key principles for creating dashboards that tell compelling data stories.",
      content: `
        <h2>Design Principles</h2>
        <p><strong>Clarity:</strong> Use clear, concise titles and labels. Avoid clutter and focus on key metrics.</p>
        <p><strong>Consistency:</strong> Maintain consistent color schemes, fonts, and layout patterns throughout your dashboard.</p>
        
        <h2>Visual Hierarchy</h2>
        <p>Use size, color, and positioning to guide the viewer's eye to the most important information first.</p>
        
        <h2>Interactive Elements</h2>
        <p>Implement filters, slicers, and drill-through functionality to allow users to explore data at different levels of detail.</p>
      `,
      date: "Dec 5, 2024",
      readTime: "10 min read", 
      category: "Visualization",
      tags: ["Power BI", "Dashboard", "Design"],
      image: powerBiImage
    },
    {
      id: 4,
      slug: "etl-best-practices-raw-data-business-insights",
      title: "ETL Best Practices: From Raw Data to Business Insights",
      excerpt: "Master the art of data transformation with proven ETL methodologies and tools.",
      content: `
        <h2>Extract Phase</h2>
        <p>Identify data sources, establish connections, and implement incremental extraction strategies to minimize processing time.</p>
        
        <h2>Transform Phase</h2>
        <p>Clean, validate, and standardize data formats. Apply business rules and calculations to create meaningful metrics.</p>
        
        <h2>Load Phase</h2>
        <p>Choose appropriate loading strategies: full load vs. incremental load based on data volume and business requirements.</p>
        
        <h2>Monitoring & Maintenance</h2>
        <p>Implement error handling, logging, and monitoring to ensure data quality and pipeline reliability.</p>
      `,
      date: "Nov 28, 2024",
      readTime: "15 min read",
      category: "Data Engineering", 
      tags: ["ETL", "Data Pipeline", "SQL"],
      image: etlImage
    },
    {
      id: 5,
      slug: "future-data-analytics-trends-2025",
      title: "The Future of Data Analytics: Trends to Watch in 2025",
      excerpt: "Discover emerging trends and technologies shaping the future of data analytics.",
      content: `
        <h2>Emerging Trends</h2>
        <p><strong>AI-Powered Analytics:</strong> Automated insights generation and natural language query capabilities.</p>
        <p><strong>Real-time Analytics:</strong> Stream processing and edge computing for instant decision-making.</p>
        
        <h2>Technology Evolution</h2>
        <p>Cloud-native analytics platforms, serverless architectures, and containerized data processing solutions.</p>
        
        <h2>Data Democratization</h2>
        <p>Self-service analytics tools that enable non-technical users to explore and analyze data independently.</p>
      `,
      date: "Nov 20, 2024",
      readTime: "6 min read",
      category: "Industry Trends",
      tags: ["Trends", "Analytics", "Future"],
      image: futureImage
    },
    {
      id: 6,
      slug: "sql-optimization-techniques-large-datasets",
      title: "SQL Optimization Techniques for Large Datasets",
      excerpt: "Proven strategies to improve query performance and handle big data efficiently.",
      content: `
        <h2>Indexing Strategies</h2>
        <p>Create appropriate indexes on frequently queried columns. Use composite indexes for multi-column searches.</p>
        
        <h2>Query Optimization</h2>
        <p>Use EXPLAIN plans to understand query execution. Avoid SELECT *, use specific column names, and optimize JOIN operations.</p>
        
        <h2>Partitioning</h2>
        <p>Implement table partitioning for large datasets to improve query performance and maintenance operations.</p>
        
        <h2>Performance Monitoring</h2>
        <p>Monitor query execution times, resource usage, and identify bottlenecks using database profiling tools.</p>
      `,
      date: "Nov 15, 2024",
      readTime: "14 min read",
      category: "Database",
      tags: ["SQL", "Performance", "Optimization"],
      image: sqlImage
    },
    {
      id: 7,
      slug: "complete-data-pipeline-tutorial",
      title: "Building a Complete Data Pipeline: From Raw Data to Interactive Dashboards",
      excerpt: "A comprehensive tutorial showing how to build an end-to-end data pipeline with code examples and visual demonstrations.",
      content: `
        <h2>Introduction</h2>
        <p>In this comprehensive tutorial, we'll build a complete data pipeline from scratch. You'll learn how to extract data from various sources, transform it using Python, and create beautiful dashboards for data visualization.</p>
        
        <img src="PIPELINE_IMAGE_PLACEHOLDER" alt="Modern data pipeline architecture showing the flow from data sources through ETL processes to analytics and dashboards" />
        
        <h2>Setting Up the Environment</h2>
        <p>First, let's set up our Python environment with the necessary libraries for our data pipeline:</p>
        
        <pre><code class="language-bash"># Install required packages
pip install pandas numpy sqlalchemy psycopg2-binary requests beautifulsoup4
pip install matplotlib seaborn plotly dash
pip install apache-airflow</code></pre>
        
        <h2>Step 1: Data Extraction</h2>
        <p>We'll start by creating functions to extract data from different sources - APIs, databases, and CSV files:</p>
        
        <pre><code class="language-python">import pandas as pd
import requests
from sqlalchemy import create_engine
import json

def extract_from_api(url, headers=None):
    """Extract data from REST API"""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API extraction failed: {e}")
        return None

def extract_from_database(connection_string, query):
    """Extract data from SQL database"""
    try:
        engine = create_engine(connection_string)
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        print(f"Database extraction failed: {e}")
        return None

def extract_from_csv(file_path):
    """Extract data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"CSV extraction failed: {e}")
        return None

# Example usage
sales_data = extract_from_csv('sales_data.csv')
customer_data = extract_from_database(
    'postgresql://user:pass@localhost/db', 
    'SELECT * FROM customers'
)
weather_data = extract_from_api('https://api.weather.com/v1/current')</code></pre>
        
        <h2>Step 2: Data Transformation</h2>
        <p>Now we'll clean and transform our data to make it analysis-ready:</p>
        
        <pre><code class="language-python">def clean_sales_data(df):
    """Clean and standardize sales data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df['revenue'] = df['revenue'].fillna(0)
    df['customer_id'] = df['customer_id'].fillna('UNKNOWN')
    
    # Standardize date formats
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    
    # Create derived columns
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['quarter'] = df['sale_date'].dt.quarter
    
    return df

def merge_datasets(sales_df, customer_df):
    """Merge sales and customer data"""
    merged_df = sales_df.merge(
        customer_df, 
        on='customer_id', 
        how='left'
    )
    
    # Calculate customer lifetime value
    clv = merged_df.groupby('customer_id')['revenue'].sum().reset_index()
    clv.columns = ['customer_id', 'lifetime_value']
    
    final_df = merged_df.merge(clv, on='customer_id')
    return final_df

# Transform the data
cleaned_sales = clean_sales_data(sales_data)
final_dataset = merge_datasets(cleaned_sales, customer_data)</code></pre>
        
        <h2>Step 3: Data Loading and Storage</h2>
        <p>Next, we'll load our transformed data into a data warehouse:</p>
        
        <pre><code class="language-python">def load_to_warehouse(df, table_name, connection_string):
    """Load data to warehouse"""
    try:
        engine = create_engine(connection_string)
        df.to_sql(
            table_name, 
            engine, 
            if_exists='replace', 
            index=False,
            method='multi',
            chunksize=1000
        )
        print(f"Successfully loaded {len(df)} rows to {table_name}")
    except Exception as e:
        print(f"Loading failed: {e}")

# Load to warehouse
warehouse_conn = 'postgresql://warehouse_user:pass@warehouse_host/analytics_db'
load_to_warehouse(final_dataset, 'sales_analytics', warehouse_conn)</code></pre>
        
        <h2>Step 4: Creating Dashboards</h2>
        <p>Finally, let's create an interactive dashboard to visualize our data:</p>
        
        <img src="DASHBOARD_IMAGE_PLACEHOLDER" alt="Professional dashboard interface showing various data visualizations including charts, metrics, and interactive filters" />
        
        <pre><code class="language-python">import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

app = Dash(__name__)

# Create visualizations
def create_revenue_chart(df):
    """Create monthly revenue trend chart"""
    monthly_revenue = df.groupby(['year', 'month'])['revenue'].sum().reset_index()
    monthly_revenue['date'] = pd.to_datetime(monthly_revenue[['year', 'month']].assign(day=1))
    
    fig = px.line(
        monthly_revenue, 
        x='date', 
        y='revenue',
        title='Monthly Revenue Trend',
        labels={'revenue': 'Revenue ($)', 'date': 'Month'}
    )
    
    fig.update_layout(
        template='plotly_white',
        height=400,
        title_font_size=16
    )
    
    return fig

def create_customer_segmentation(df):
    """Create customer segmentation chart"""
    segments = df.groupby('customer_segment')['lifetime_value'].agg(['count', 'mean']).reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=segments['customer_segment'],
            y=segments['count'],
            name='Customer Count',
            yaxis='y'
        ),
        go.Scatter(
            x=segments['customer_segment'],
            y=segments['mean'],
            name='Average CLV',
            yaxis='y2',
            mode='lines+markers'
        )
    ])
    
    fig.update_layout(
        title='Customer Segmentation Analysis',
        yaxis=dict(title='Number of Customers'),
        yaxis2=dict(title='Average CLV ($)', overlaying='y', side='right'),
        template='plotly_white'
    )
    
    return fig

# Dashboard layout
app.layout = html.Div([
    html.H1('Sales Analytics Dashboard', className='text-center'),
    
    dcc.Graph(
        id='revenue-chart',
        figure=create_revenue_chart(final_dataset)
    ),
    
    dcc.Graph(
        id='segmentation-chart',
        figure=create_customer_segmentation(final_dataset)
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)</code></pre>
        
        <h2>Step 5: Automation with Apache Airflow</h2>
        <p>To make our pipeline production-ready, let's create an Airflow DAG for automation:</p>
        
        <pre><code class="language-python">from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'sales_data_pipeline',
    default_args=default_args,
    description='Complete sales data pipeline',
    schedule_interval='@daily',
    catchup=False
)

def run_extraction():
    """Run data extraction tasks"""
    # Your extraction code here
    pass

def run_transformation():
    """Run data transformation tasks"""
    # Your transformation code here
    pass

def run_loading():
    """Run data loading tasks"""
    # Your loading code here
    pass

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=run_extraction,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=run_transformation,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=run_loading,
    dag=dag
)

# Set task dependencies
extract_task >> transform_task >> load_task</code></pre>
        
        <h2>Best Practices and Tips</h2>
        <p><strong>Data Quality:</strong> Always implement data validation checks at each stage of your pipeline.</p>
        <p><strong>Error Handling:</strong> Use try-catch blocks and implement proper logging for debugging.</p>
        <p><strong>Scalability:</strong> Design your pipeline to handle increasing data volumes.</p>
        <p><strong>Monitoring:</strong> Set up alerts for pipeline failures and data quality issues.</p>
        
        <h2>Conclusion</h2>
        <p>Building a complete data pipeline requires careful planning and implementation of each component. By following this tutorial, you now have a solid foundation for creating robust, scalable data pipelines that can handle real-world business requirements.</p>
        
        <p>The combination of proper data extraction, transformation, loading, and visualization creates a powerful system that can drive data-driven decision making in any organization.</p>
      `,
      date: "Jan 2, 2025",
      readTime: "20 min read",
      category: "Data Engineering",
      tags: ["Python", "ETL", "Dashboard", "Airflow", "Pipeline"],
      image: dataPipelineImage
    }
  ];

  const blog = blogs.find(b => b.slug === slug);

  if (!blog) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="pt-24 pb-20">
          <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h1 className="text-4xl font-bold mb-4">Blog Post Not Found</h1>
            <p className="text-muted-foreground mb-8">The blog post you're looking for doesn't exist.</p>
            <Button onClick={() => {
              navigate('/');
              setTimeout(() => {
                const blogsSection = document.getElementById('blogs');
                if (blogsSection) {
                  blogsSection.scrollIntoView({ behavior: 'smooth' });
                }
              }, 100);
            }}>
              Back to Blogs
            </Button>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <article className="pt-24 pb-20">
        <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Back Button */}
          <Button 
            variant="ghost" 
            onClick={() => {
              navigate('/');
              setTimeout(() => {
                const blogsSection = document.getElementById('blogs');
                if (blogsSection) {
                  blogsSection.scrollIntoView({ behavior: 'smooth' });
                }
              }, 100);
            }}
            className="mb-8 text-muted-foreground hover:text-foreground"
          >
            ← Back to Blogs
          </Button>

          {/* SEO-Optimized Article Header */}
          <article className="mb-8">
            <header>
              <img 
                src={blog.image} 
                alt={`${blog.title} - Data Analytics and Business Intelligence insights by expert data analyst Obed Makori`}
                className="w-full h-64 object-cover rounded-lg mb-4"
              />
              <h1 className="text-4xl lg:text-5xl font-bold mb-4 text-center">
                {blog.title}
              </h1>
              <div className="flex items-center justify-center gap-4 text-muted-foreground mb-4">
                <span>By Obed Makori - Expert Data Analyst & BI Developer</span>
                <span>•</span>
                <time dateTime={blog.date}>{blog.date}</time>
                <span>•</span>
                <span>{blog.readTime}</span>
                <span>•</span>
                <Badge variant="secondary" className="bg-primary/10 text-primary">
                  {blog.category}
                </Badge>
              </div>
            </header>
          </article>

          {/* Article Content */}
          <Card>
            <CardContent className="p-8">
              <BlogContent content={blog.content} />
              
              {/* Tags */}
              <div className="flex flex-wrap gap-2 mt-8 pt-8 border-t border-border">
                {blog.tags.map((tag) => (
                  <Badge 
                    key={tag} 
                    variant="outline" 
                    className="text-sm border-muted hover:border-primary hover:text-primary transition-colors duration-200"
                  >
                    {tag}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Comments Section */}
          <CommentsSection blogSlug={blog.slug} />
        </div>
      </article>
      <Footer />
    </div>
  );
};

export default BlogDetail;