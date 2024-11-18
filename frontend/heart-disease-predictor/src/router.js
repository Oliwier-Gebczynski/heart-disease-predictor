import { createRouter, createWebHistory } from 'vue-router';
import LoginPanel from './components/LoginPanel.vue';
import TestConnection from './App.vue';

const routes = [
  {
    path: '/',
    name: 'LoginPanel',
    component: LoginPanel,
  },
  {
    path: '/test-connection',
    name: 'TestConnection',
    component: TestConnection,
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
