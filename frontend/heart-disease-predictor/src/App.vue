<template>
  <div id="app">
    <h1>Test Connection</h1>
    <button @click="testConnection">Test Connection to Backend</button>
  </div>
</template>

<script>
export default {
  name: 'App',
  methods: {
    async testConnection() {
      try {
        const response = await fetch('http://localhost:3000/test');
        const data = await response.json();
        console.log('Response from backend:', data.message);
      } catch (error) {
        console.error('Error connecting to backend:', error);
      }
    }
  }
}
</script>

<style>
#app {
  text-align: center;
  margin-top: 50px;
}
</style>
