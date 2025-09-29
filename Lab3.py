"""
Simulación de Colas en Supermercado (versión mejorada)
Universidad de Panamá - Facultad de Informática, Electrónica y Comunicación
Materia: Modelos y Simulación

Modificaciones solicitadas:
 - Cambié las distribuciones exponenciales por normales o uniformes (configurable).
 - Agregué un sistema de prioridades donde algunos clientes (e.g., prioridad alta) son atendidos antes.
 - Horarios variables: la intensidad de llegada cambia según franjas horarias (picos y valles).
 - Cajeras con diferentes velocidades: cada cajera tiene parámetros de servicio propios.

"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import random

# -----------------------------
# Clases y funciones principales
# -----------------------------

class Cashier:
    """Encapsula un recurso de cajera con parámetros de servicio distintos.
    Usa simpy.PriorityResource para permitir prioridades en las solicitudes.
    """
    def __init__(self, env, id, service_mean=5.0, service_std=1.0, dist='normal'):
        self.env = env
        self.id = id
        self.resource = simpy.PriorityResource(env, capacity=1)
        self.service_mean = service_mean
        self.service_std = service_std
        self.dist = dist  # 'normal' o 'uniform'
        self.busy_time = 0.0

    def sample_service_time(self):
        if self.dist == 'normal':
            # Truncar para evitar negativos
            t = np.random.normal(self.service_mean, self.service_std)
            return max(0.01, t)
        elif self.dist == 'uniform':
            # Uniform entre mean-std y mean+std
            low = max(0.01, self.service_mean - self.service_std)
            high = max(low + 0.01, self.service_mean + self.service_std)
            return np.random.uniform(low, high)
        else:
            # fallback
            return max(0.01, np.random.normal(self.service_mean, self.service_std))

class SupermarketSimulation:
    """Clase para simular el sistema de colas del supermercado con prioridades,
    franjas horarias y cajeras heterogéneas.
    """
    def __init__(self,
                 num_cajeras=1,
                 tiempo_simulacion=60,
                 arrival_distribution='normal',
                 priority_prob=0.2,
                 franjas_horario=None,
                 cajeras_params=None,
                 random_seed=None):
        self.num_cajeras = num_cajeras
        self.tiempo_simulacion = tiempo_simulacion
        self.arrival_distribution = arrival_distribution  # 'normal' o 'uniform'
        self.priority_prob = priority_prob  # probabilidad de que un cliente sea prioridad alta

        # franjas_horario: lista de dicts {"start":0, "end":30, "mean_interarrival":4.0, "sd":1.0 }
        # si None, se usa una franja uniforme con mean=4
        if franjas_horario is None:
            self.franjas_horario = [{'start': 0, 'end': tiempo_simulacion, 'mean_interarrival': 4.0, 'sd': 1.0}]
        else:
            self.franjas_horario = franjas_horario

        # cajeras_params: lista de dicts por cajera: {"service_mean":5.0, "service_std":1.0, "dist":"normal"}
        if cajeras_params is None:
            cajeras_params = [{'service_mean':5.0, 'service_std':1.0, 'dist':'normal'} for _ in range(num_cajeras)]
        self.cajeras_params = cajeras_params

        # Estadísticas
        self.clientes_atendidos = 0
        self.tiempos_espera = []
        self.tiempos_sistema = []
        self.utilizacion_cajeras = defaultdict(float)
        self.cola_maxima = 0
        self.detalle_clientes = []  # para análisis más fino

        # Random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

    def _get_current_franja(self, now):
        for f in self.franjas_horario:
            if f['start'] <= now < f['end']:
                return f
        return self.franjas_horario[-1]

    def _sample_interarrival(self, now):
        f = self._get_current_franja(now)
        mean = f.get('mean_interarrival', 4.0)
        sd = f.get('sd', max(0.1, 0.2 * mean))
        if self.arrival_distribution == 'normal':
            t = np.random.normal(mean, sd)
            return max(0.01, t)
        elif self.arrival_distribution == 'uniform':
            low = max(0.01, mean - sd)
            high = max(low + 0.01, mean + sd)
            return np.random.uniform(low, high)
        else:
            # default: normal
            t = np.random.normal(mean, sd)
            return max(0.01, t)

    def cliente_llegada(self, env, cashiers, cliente_id, prioridad):
        """Proceso de llegada y atención de un cliente con prioridad.
        prioridad: entero (0 = alta, 1 = normal). En simpy.PriorityResource, menor valor -> mayor prioridad.
        """
        tiempo_llegada = env.now
        # Registrar tamaño de colas para todas las cajeras y calcular cola mínima
        lengths = [len(c.resource.queue) + (1 if c.resource.count == c.resource.capacity else 0) for c in cashiers]
        cajera_elegida = cashiers[int(np.argmin(lengths))]

        tamaño_cola = lengths[cashiers.index(cajera_elegida)]
        if tamaño_cola > self.cola_maxima:
            self.cola_maxima = tamaño_cola

        # Solicitar servicio con prioridad
        with cajera_elegida.resource.request(priority=prioridad) as req:
            yield req
            tiempo_inicio_servicio = env.now
            tiempo_espera = tiempo_inicio_servicio - tiempo_llegada
            self.tiempos_espera.append(tiempo_espera)

            # Muestreo de tiempo de servicio según la cajera elegida
            tiempo_atencion = cajera_elegida.sample_service_time()
            # Registrar que la cajera está ocupada por este tiempo
            self.utilizacion_cajeras[cajera_elegida.id] += tiempo_atencion

            # Guardar detalle para análisis
            self.detalle_clientes.append({
                'id': cliente_id,
                'llegada': tiempo_llegada,
                'inicio_servicio': tiempo_inicio_servicio,
                'salida': tiempo_inicio_servicio + tiempo_atencion,
                'espera': tiempo_espera,
                'total_en_sistema': tiempo_atencion + tiempo_espera,
                'prioridad': prioridad,
                'cajera_id': cajera_elegida.id
            })

            yield env.timeout(tiempo_atencion)

            tiempo_salida = env.now
            tiempo_sistema = tiempo_salida - tiempo_llegada
            self.tiempos_sistema.append(tiempo_sistema)
            self.clientes_atendidos += 1

    def generador_clientes(self, env, cashiers):
        cliente_id = 1
        while True:
            now = env.now
            # Sacar tiempo hasta la siguiente llegada según franjas/ distrib
            t_entre = self._sample_interarrival(now)
            yield env.timeout(t_entre)

            # Determinar prioridad aleatoria (0 alta, 1 normal)
            prioridad = 0 if random.random() < self.priority_prob else 1
            env.process(self.cliente_llegada(env, cashiers, cliente_id, prioridad))
            cliente_id += 1

    def ejecutar_simulacion(self):
        env = simpy.Environment()
        # Crear objetos Cashier
        cashiers = []
        for i in range(self.num_cajeras):
            params = self.cajeras_params[i] if i < len(self.cajeras_params) else self.cajeras_params[-1]
            c = Cashier(env, id=i, service_mean=params.get('service_mean',5.0),
                       service_std=params.get('service_std',1.0), dist=params.get('dist','normal'))
            cashiers.append(c)

        env.process(self.generador_clientes(env, cashiers))
        env.run(until=self.tiempo_simulacion)

        # Calcular utilización porcentual
        utiliz_percent = {}
        for i in range(self.num_cajeras):
            utiliz_percent[i] = (self.utilizacion_cajeras[i] / self.tiempo_simulacion) * 100
        self.utilizacion_cajeras = utiliz_percent

        # Convertir detalle clientes a DataFrame para análisis opcional
        if self.detalle_clientes:
            self.df_detalle = pd.DataFrame(self.detalle_clientes)
        else:
            self.df_detalle = pd.DataFrame()

    def generar_reporte(self):
        print(f"\n{'='*60}")
        print(f"REPORTE DE SIMULACIÓN - {self.num_cajeras} CAJERA(S) - {self.tiempo_simulacion} MINUTOS")
        print(f"{'='*60}")

        if self.tiempos_espera:
            print(f"Clientes atendidos: {self.clientes_atendidos}")
            print(f"Tiempo promedio de espera: {np.mean(self.tiempos_espera):.2f} minutos")
            print(f"Tiempo máximo de espera: {np.max(self.tiempos_espera):.2f} minutos")
            print(f"Tiempo promedio en el sistema: {np.mean(self.tiempos_sistema):.2f} minutos")
            print(f"Cola máxima observada: {self.cola_maxima} clientes")
            for i in range(self.num_cajeras):
                print(f"Utilización Cajera {i+1}: {self.utilizacion_cajeras.get(i,0):.1f}%")

            # Mostrar desagregado por prioridad
            df = self.df_detalle
            if not df.empty:
                for p in sorted(df['prioridad'].unique()):
                    sub = df[df['prioridad'] == p]
                    print(f"Prioridad {p} - N={len(sub)} - Espera media: {sub['espera'].mean():.2f} min")
        else:
            print("No se atendieron clientes en este período")

        return {
            'clientes_atendidos': self.clientes_atendidos,
            'tiempo_espera_promedio': np.mean(self.tiempos_espera) if self.tiempos_espera else 0,
            'tiempo_espera_maximo': np.max(self.tiempos_espera) if self.tiempos_espera else 0,
            'tiempo_sistema_promedio': np.mean(self.tiempos_sistema) if self.tiempos_sistema else 0,
            'utilizacion_cajeras': dict(self.utilizacion_cajeras),
            'cola_maxima': self.cola_maxima,
            'detalle_df': getattr(self, 'df_detalle', pd.DataFrame())
        }

# -----------------------------
# Funciones de ayuda: escenarios y gráficas
# -----------------------------

def comparar_escenarios():
    print("INICIANDO SIMULACIONES COMPARATIVAS")
    print("="*60)

    # Semilla para reproducibilidad
    seed = 12345

    # Escenario A: 1 cajera, 60 min, llegadas normales con variabilidad diurna (pico corto)
    franjas = [
        {'start':0, 'end':20, 'mean_interarrival':6.0, 'sd':1.0},  # baja
        {'start':20, 'end':40, 'mean_interarrival':3.0, 'sd':0.8}, # pico
        {'start':40, 'end':60, 'mean_interarrival':5.0, 'sd':1.0}  # media
    ]
    caj_params_1 = [{'service_mean':5.0, 'service_std':1.0, 'dist':'normal'}]

    print('\n🔹 CASO A: 1 CAJERA - 60 MINUTOS - LLEGADAS NORMALES CON PICO')
    simA = SupermarketSimulation(num_cajeras=1, tiempo_simulacion=60,
                                  arrival_distribution='normal', priority_prob=0.2,
                                  franjas_horario=franjas, cajeras_params=caj_params_1, random_seed=seed)
    simA.ejecutar_simulacion()
    resA = simA.generar_reporte()

    # Escenario B: 1 cajera, 60 min, llegadas uniformes (menos variabilidad)
    print('\n🔹 CASO B: 1 CAJERA - 60 MINUTOS - LLEGADAS UNIFORMES')
    caj_params_1u = [{'service_mean':5.0, 'service_std':2.0, 'dist':'uniform'}]
    simB = SupermarketSimulation(num_cajeras=1, tiempo_simulacion=60,
                                  arrival_distribution='uniform', priority_prob=0.15,
                                  franjas_horario=[{'start':0,'end':60,'mean_interarrival':4.0,'sd':1.5}],
                                  cajeras_params=caj_params_1u, random_seed=seed)
    simB.ejecutar_simulacion()
    resB = simB.generar_reporte()

    # Escenario C: 2 cajeras heterogéneas
    print('\n🔹 CASO C: 2 CAJERAS - 60 MINUTOS - CAJERAS CON DISTINTA VELOCIDAD')
    caj_params_2 = [
        {'service_mean':4.0, 'service_std':0.8, 'dist':'normal'},  # cajera rápida
        {'service_mean':6.0, 'service_std':1.2, 'dist':'normal'}   # cajera lenta
    ]
    simC = SupermarketSimulation(num_cajeras=2, tiempo_simulacion=60,
                                  arrival_distribution='normal', priority_prob=0.25,
                                  franjas_horario=franjas, cajeras_params=caj_params_2, random_seed=seed)
    simC.ejecutar_simulacion()
    resC = simC.generar_reporte()

    # Graficar comparativos
    crear_graficos_comparativos([resA, resB, resC])

    return resA, resB, resC


def crear_graficos_comparativos(resultados):
    escenarios = ['1 Cajera - Pico', '1 Cajera - Uniforme', '2 Cajeras - Heterog.']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Clientes atendidos
    clientes = [r['clientes_atendidos'] for r in resultados]
    ax1.bar(escenarios, clientes)
    ax1.set_title('Clientes Atendidos por Escenario')
    ax1.set_ylabel('Número de Clientes')
    ax1.tick_params(axis='x', rotation=30)

    # Tiempo promedio de espera
    tiempos_espera = [r['tiempo_espera_promedio'] for r in resultados]
    ax2.bar(escenarios, tiempos_espera)
    ax2.set_title('Tiempo Promedio de Espera')
    ax2.set_ylabel('Minutos')
    ax2.tick_params(axis='x', rotation=30)

    # Cola máxima
    colas_max = [r['cola_maxima'] for r in resultados]
    ax3.bar(escenarios, colas_max)
    ax3.set_title('Cola Máxima Observada')
    ax3.set_ylabel('Número de Clientes')
    ax3.tick_params(axis='x', rotation=30)

    # Utilización promedio (promediando cajeras si aplica)
    utilizaciones = []
    for r in resultados:
        vals = list(r['utilizacion_cajeras'].values())
        utilizaciones.append(np.mean(vals) if vals else 0)

    ax4.bar(escenarios, utilizaciones)
    ax4.set_title('Utilización Promedio de Cajeras (%)')
    ax4.set_ylabel('Porcentaje (%)')
    ax4.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()


def simulacion_sensibilidad(max_cajeras=4):
    print('\n' + '='*60)
    print('ANÁLISIS DE SENSIBILIDAD: VARIANDO NÚMERO DE CAJERAS')
    print('='*60)

    resultados = []
    for n in range(1, max_cajeras+1):
        # Crear cajeras heterogéneas de ejemplo: cada cajera i más lenta en promedio
        caj_params = []
        for i in range(n):
            caj_params.append({'service_mean':5.0 + 0.5*i, 'service_std':1.0 + 0.1*i, 'dist':'normal'})

        sim = SupermarketSimulation(num_cajeras=n, tiempo_simulacion=120,
                                    arrival_distribution='normal', priority_prob=0.2,
                                    franjas_horario=[{'start':0,'end':120,'mean_interarrival':4.0,'sd':1.0}],
                                    cajeras_params=caj_params, random_seed=2025)
        sim.ejecutar_simulacion()
        r = sim.generar_reporte()
        r['num_cajeras'] = n
        resultados.append(r)

    df = pd.DataFrame(resultados)
    print('\nTabla de Sensibilidad:')
    print(df[['num_cajeras', 'clientes_atendidos', 'tiempo_espera_promedio', 'cola_maxima']])
    return df

# -----------------------------
# Punto de entrada
# -----------------------------
if __name__ == '__main__':
    print('LABORATORIO: SIMULACIÓN DE COLAS EN SUPERMERCADO - VERSIÓN MEJORADA')
    print('Universidad de Panamá - FIEC')
    print('='*60)

    # Ejecutar comparativo
    resultados = comparar_escenarios()

    # Sensibilidad
    df_sens = simulacion_sensibilidad(max_cajeras=4)

    print('\n' + '='*60)
    print('SIMULACIÓN COMPLETADA - Revise gráficos y resultados')
    print('='*60)
