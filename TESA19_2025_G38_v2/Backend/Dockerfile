# syntax=docker/dockerfile:1.7
# ใช้ Node.js base image
FROM node:20

# ตั้ง working directory ใน container
WORKDIR /usr/src/app

# คัดลอกไฟล์ package.json และ package-lock.json ก่อน
COPY package*.json ./

# ติดตั้ง dependencies
RUN --mount=type=cache,target=/root/.npm \
	sh -c "npm ci --omit=dev || npm install --omit=dev"

# คัดลอกไฟล์ทั้งหมดเข้ามา
COPY . .

# เปิดพอร์ตที่ backend ใช้งาน
EXPOSE 3000

# รันเซิร์ฟเวอร์
CMD ["node", "server.js"]
