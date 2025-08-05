import pandas as pd
# sql_result = [
#     {"Name": "John Doe", "Description": "John is a software engineer at OpenAI in San Francisco."},
#     {"Name": "Jane Smith", "Description": "Jane works for Google and lives in New York."},
# ]
# df=pd.DataFrame(sql_result)
# print(df)
df=pd.read_excel("SO(real).xlsx")
result=df.head(2).to_dict(orient='records')
# print(result)

sql_result = [
        {
            "name": "Alice Johnson",
            "job_description": "Alice is a data analyst at Meta, working on user behavior analytics.",
            "comments": "She recently presented her findings in a global tech conference in Paris.",
            "reasons": "Her deep knowledge of behavioral patterns makes her suitable for customer insight projects."
        },
        {
            "name": "Bob Martinez",
            "job_description": "Bob manages cloud infrastructure for Microsoft in Seattle.",
            "comments": "Handled multiple outages without service disruptions.",
            "reasons": "His experience in system resiliency and scaling is unmatched."
        },
        {
            "name": "Carla Singh",
            "job_description": "Carla works as a security consultant at IBM based in Toronto.",
            "comments": "Suggested vital policy updates after the internal audit.",
            "reasons": "Proactive approach to vulnerabilities and quick response time."
        },
        {
            "name": "Daniel Lee",
            "job_description": "Daniel is a backend developer at Netflix handling video optimization services.",
            "comments": "Led the re-architecture of the media pipeline.",
            "reasons": "Critical for improving performance across global CDN nodes."
        },
        {
            "name": "Emily Zhang",
            "job_description": "Emily leads the AI research team at Amazon Web Services in Beijing.",
            "comments": "Published several papers on neural architecture search.",
            "reasons": "Excellent fit for LLM deployment in production environments."
        },
        {
            "name": "Frank Muller",
            "job_description": "Frank is a mobile application tester working remotely for Spotify.",
            "comments": "Found and reported over 250 unique bugs last quarter.",
            "reasons": "Strong QA background and familiarity with agile teams."
        },
        {
            "name": "Grace Park",
            "job_description": "Grace designs UX workflows at Adobe’s San Francisco studio.",
            "comments": "Helped redesign the Creative Cloud onboarding experience.",
            "reasons": "User-focused thinker with strong visual communication skills."
        },
        {
            "name": "Henry Kim",
            "job_description": "Henry develops APIs for Salesforce CRM systems in Seoul.",
            "comments": "Collaborated with marketing to add personalization APIs.",
            "reasons": "Knows the product well and interacts with clients frequently."
        },
        {
            "name": "Irene Blake",
            "job_description": "Irene is a cybersecurity lead at Oracle, focusing on identity management.",
            "comments": "Pioneered zero-trust initiatives within the department.",
            "reasons": "Leads by example and keeps systems compliant."
        },
        {
            "name": "Jason Patel",
            "job_description": "Jason works on DevOps automation at Dropbox in Austin.",
            "comments": "Built reusable pipelines with Terraform and GitHub Actions.",
            "reasons": "Ideal for infrastructure-as-code standardization."
        },
        {
            "name": "Kavya Rao",
            "job_description": "Kavya is a machine learning engineer at Nvidia working on generative models.",
            "comments": "Ran successful experiments on diffusion-based text-to-image systems.",
            "reasons": "Her publications are referenced by peers worldwide."
        },
        {
            "name": "Liam Wilson",
            "job_description": "Liam is a project coordinator for Capgemini, based in London.",
            "comments": "Ensured cross-regional teams hit all delivery milestones.",
            "reasons": "Strong in follow-through and logistics."
        },
        {
            "name": "Maria Gonzalez",
            "job_description": "Maria is a senior QA engineer for Cisco networking devices.",
            "comments": "Created test cases that reduced bugs in firmware by 40%.",
            "reasons": "Excellent eye for detail and strong scripting skills."
        },
        {
            "name": "Nathan Silva",
            "job_description": "Nathan is a blockchain developer at ConsenSys in Lisbon.",
            "comments": "Launched a private Ethereum testnet for clients.",
            "reasons": "Hands-on with decentralized systems and security."
        },
        {
            "name": "Olivia Chen",
            "job_description": "Olivia leads customer experience design for Airbnb in Singapore.",
            "comments": "Redefined host onboarding flows, improving retention.",
            "reasons": "Balances aesthetics with usability in high-impact products."
        },
        {
            "name": "Pranav Mehta",
            "job_description": "Pranav is a data privacy consultant for Deloitte in Mumbai.",
            "comments": "Drafted GDPR and HIPAA-compliant frameworks for a major bank.",
            "reasons": "Deep legal-tech crossover knowledge."
        },
        {
            "name": "Quinn Bailey",
            "job_description": "Quinn writes internal tools for Atlassian teams in Sydney.",
            "comments": "Simplified admin dashboards for faster troubleshooting.",
            "reasons": "Loves automation and reducing operational overhead."
        },
        {
            "name": "Rita Kapoor",
            "job_description": "Rita is a full-stack developer for Shopify's analytics team.",
            "comments": "Upgraded legacy reporting systems using GraphQL.",
            "reasons": "Good balance between backend depth and frontend polish."
        },
        {
            "name": "Samir Khan",
            "job_description": "Samir is a product manager at Zoom, focused on user engagement.",
            "comments": "Led integrations with Google Calendar and Slack.",
            "reasons": "Understands user habits and enterprise use cases."
        },
        {
            "name": "Tina Rossi",
            "job_description": "Tina handles legal documentation for Accenture's tech contracts.",
            "comments": "Streamlined vendor approvals across multiple departments.",
            "reasons": "Critical for scaling without bottlenecks."
        }
    ]
df=pd.DataFrame(sql_result)
print(df)