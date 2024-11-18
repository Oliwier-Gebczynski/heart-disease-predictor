<template>
  <div class="login-container">
    <h1 class="login-container__title">Heart Disease Predictor</h1>
    <form @submit.prevent="login" class="login-container__form">
      <div class="login-container__form-group">
        <label for="username" class="login-container__form-group-label">Username</label>
        <input
          type="text"
          id="username"
          v-model="username"
          class="login-container__form-group-input"
          placeholder="Enter your username"
          required
        />
      </div>
      <div class="login-container__form-group">
        <label for="password" class="login-container__form-group-label">Password</label>
        <input
          type="password"
          id="password"
          v-model="password"
          class="login-container__form-group-input"
          placeholder="Enter your password"
          required
        />
      </div>
      <button type="submit" class="login-container__button">Login</button>
    </form>
    <p v-if="errorMessage" class="login-container__error-message">
      {{ errorMessage }}
    </p>
  </div>
</template>

<script>
export default {
  name: "LoginPanel",
  data() {
    return {
      username: "",
      password: "",
      errorMessage: "",
    };
  },
  methods: {
    async login() {
      try {
        const response = await fetch("http://localhost:3000/login", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            username: this.username,
            password: this.password,
          }),
        });

        if (!response.ok) {
          throw new Error("Invalid credentials");
        }

        const data = await response.json();
        this.redirectBasedOnRole(data.role);
      } catch (error) {
        this.errorMessage = error.message || "An error occurred";
      }
    },
    redirectBasedOnRole(role) {
      switch (role) {
        case "Admin":
          this.$router.push("/admin");
          break;
        case "Doctor":
          this.$router.push("/doctor");
          break;
        case "Patient":
          this.$router.push("/patient");
          break;
        default:
          this.errorMessage = "Invalid role received";
      }
    },
  },
};
</script>

<style scoped>
.login-container {
  max-width: 400px;
  margin: 50px auto;
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 10px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.login-container__title {
  text-align: center;
  color: #d9534f;
  font-family: 'Arial', sans-serif;
  margin-bottom: 20px;
}

.login-container__form-group {
  margin-bottom: 15px;
}

.login-container__form-group-label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.login-container__form-group-input {
  width: 100%;
  padding: 10px;
  font-size: 14px;
  border: 1px solid #ddd;
  border-radius: 5px;
}

.login-container__form-group-input:focus {
  border-color: #d9534f;
  outline: none;
  box-shadow: 0 0 5px rgba(217, 83, 79, 0.5);
}

.login-container__button {
  width: 100%;
  padding: 10px;
  background-color: #d9534f;
  color: white;
  border: none;
  border-radius: 5px;
  font-size: 16px;
  cursor: pointer;
}

.login-container__button:hover {
  background-color: #c9302c;
}

.login-container__error-message {
  color: #c9302c;
  margin-top: 15px;
  text-align: center;
}
</style>
