FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install cors

COPY . .

EXPOSE 3000

CMD ["node", "index.js"]
