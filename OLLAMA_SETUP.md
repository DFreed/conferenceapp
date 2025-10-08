# Ollama Setup for AWS Amplify Deployment

This guide shows you how to run your Conference App with Ollama in a cloud environment.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐
│   AWS Amplify   │    │   AWS EC2/ECS   │
│   (Streamlit)   │───▶│   (Ollama)      │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

## Option 1: AWS EC2 with Docker (Recommended)

### Step 1: Launch EC2 Instance

1. **Launch EC2 Instance**:
   - Instance Type: `t3.large` or larger (Ollama needs memory)
   - AMI: Ubuntu 22.04 LTS
   - Storage: 20GB+ (models are large)
   - Security Group: Allow inbound on port 11434

2. **Connect to your instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

### Step 2: Install Docker and Ollama

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again to apply docker group
exit
# SSH back in
```

### Step 3: Deploy Ollama

```bash
# Create directory for your app
mkdir conference-app
cd conference-app

# Copy the docker-compose.yml file to your EC2 instance
# (You can use scp or create it manually)

# Start Ollama
docker-compose up -d

# Pull a model (this will take a while)
docker exec ollama ollama pull llama3.1:8b

# Verify it's working
curl http://localhost:11434/api/tags
```

### Step 4: Configure Amplify

1. **Set Environment Variable in Amplify**:
   - Go to your Amplify app console
   - Environment variables → Add variable
   - Key: `OLLAMA_URL`
   - Value: `http://your-ec2-public-ip:11434`

2. **Update Security Group**:
   - Make sure your EC2 security group allows inbound traffic on port 11434 from anywhere (0.0.0.0/0)
   - Or restrict to Amplify's IP ranges

## Option 2: AWS ECS with Fargate

### Step 1: Create ECS Task Definition

```json
{
  "family": "ollama-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::YOUR-ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "ollama",
      "image": "ollama/ollama:latest",
      "portMappings": [
        {
          "containerPort": 11434,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OLLAMA_HOST",
          "value": "0.0.0.0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ollama",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Step 2: Create ECS Service

1. Create ECS cluster
2. Create service with the task definition
3. Configure Application Load Balancer
4. Set `OLLAMA_URL` in Amplify to your ALB endpoint

## Option 3: Use Existing Ollama Services

### RunPod
- Sign up at runpod.io
- Deploy Ollama template
- Get the public URL
- Set as `OLLAMA_URL` in Amplify

### Modal
- Use Modal's Ollama template
- Deploy and get endpoint
- Set as `OLLAMA_URL` in Amplify

## Testing Your Setup

1. **Deploy your app to Amplify** with the `OLLAMA_URL` environment variable
2. **Test the connection** using the "Test Ollama Connection" button in your app
3. **Try a query** to see if LLM features work

## Cost Considerations

- **EC2 t3.large**: ~$60/month
- **ECS Fargate**: ~$50-100/month depending on usage
- **RunPod**: Pay-per-use, typically $0.20-0.50/hour
- **Modal**: Pay-per-use, typically $0.10-0.30/hour

## Security Notes

- Consider using VPC peering between Amplify and your Ollama instance
- Use IAM roles for authentication
- Consider adding API keys or basic auth to Ollama
- Monitor usage and costs

## Troubleshooting

1. **Connection refused**: Check security groups and firewall rules
2. **Model not found**: Make sure you've pulled the model with `ollama pull model-name`
3. **Out of memory**: Use a larger instance or smaller model
4. **Slow responses**: Consider using a GPU instance or smaller model

## Recommended Models for Production

- **Fast & Small**: `qwen2.5:3b-instruct` (3GB)
- **Balanced**: `llama3.1:8b` (4.7GB)
- **High Quality**: `llama3.1:70b` (40GB) - requires large instance

Start with a smaller model and scale up as needed!
