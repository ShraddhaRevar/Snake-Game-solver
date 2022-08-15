import contextlib
import sys, time, random
from operator import add, sub
from dataclasses import dataclass
from itertools import product
from typing import Tuple

with contextlib.redirect_stdout(None):
    import pygame
    from pygame.locals import *
from heapq import *

W = (255, 255, 255)
B = (0, 0, 0)
R = (255, 0, 0)
G = (0, 255, 0)
B = (0, 0, 255)
DG = (40, 40, 40)


@dataclass
class Base:
    Block_size: int = 20
    Block_w: int = 10
    Block_h: int = 10
    window_width = Block_size * Block_w
    window_height = Block_size * Block_h

    @staticmethod
    def node_sub(node_a: Tuple[int, int], node_b: Tuple[int, int]):
        result: Tuple[int, int] = tuple(map(sub, node_a, node_b))
        return result

    @staticmethod
    def mean(l):
        return round(sum(l) / len(l), 4)  

    @staticmethod
    def node_add(node_a: Tuple[int, int], node_b: Tuple[int, int]):
        result: Tuple[int, int] = tuple(map(add, node_a, node_b))
        return result



    @staticmethod
    def mean(l):
        return round(sum(l) / len(l), 4)


def heuristic(start, goal):
    return (start[0] - goal[0])**2 + (start[1] - goal[1])**2


class Food(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location = None

    def refresh(self, snake):
        #Putting food randomly
        available_positions = set(product(range(self.Block_w - 1), range(self.Block_h - 1))) - set(snake.body)
        #If no position is left for food then don't place food
        location = random.sample(available_positions, 1)[0] if available_positions else (-1, -1)
        self.location = location


class Snake(Base):
    def __init__(self, initial_length: int = 3, body: list = None, **kwargs):    
        super().__init__(**kwargs)
        self.initial_length = initial_length
        self.score = 0
        self.is_dead = False
        self.eaten = False
        if body:
            self.body = body
        else:
            if not 0 < initial_length < self.Block_w:
                raise ValueError(f"Initial_length should fall in (0, {self.Block_w})")

            start_x = self.Block_w // 2
            start_y = self.Block_h // 2

            start_body_x = [start_x] * initial_length
            start_body_y = range(start_y, start_y - initial_length, -1)

            self.body = list(zip(start_body_x, start_body_y))

    def get_head(self):
        return self.body[-1]

    def dead_checking(self, head, check=False):
        #Looking for a dead snake and returning a boolean
        x, y = head
        if not 0 <= x < self.Block_w or not 0 <= y < self.Block_h or head in self.body[1:]:
            if not check:
                self.is_dead = True
            return True
        return False

    def cut_tail(self):
        self.body.pop(0)

    def move(self, new_head: tuple, food: Food):
        # Check if the food is eaten by the snake (location will be same as head )
        if new_head is None:
            self.is_dead = True
            return
        if self.dead_checking(head=new_head):
            return
        self.last_direction = self.node_sub(new_head, self.get_head())
        self.body.append(new_head)
        # Additon of score 
        if self.get_head() == food.location:
            self.eaten = True
            self.score += 1
        # Otherwise, cut the tail so that snake moves forward without growing
        else:
            self.eaten = False
            self.cut_tail()


class Player(Base):
    def __init__(self, snake: Snake, food: Food, **kwargs):
        super().__init__(**kwargs)
        self.snake = snake
        self.food = food

    def _get_neighbors(self, node):
        """
        fetch and yield the four neighbours of a node
        :param node: (node_x, node_y)
        """
        for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yield self.node_add(node, diff)

    @staticmethod
    def is_node_in_queue(node: tuple, queue: iter):
        """
        Check if element is in a nested list
        """
        return any(node in sublist for sublist in queue)

    def Check_Move(self, node: tuple, snake: Snake):
        # this is cheking the valid move and returning boolean
        x, y = node
        if not 0 <= x < self.Block_w or not 0 <= y < self.Block_h or node in snake.body:
            return True
        return False


class BFS(Player):
    def __init__(self, snake: Snake, food: Food, **kwargs):
        super().__init__(snake=snake, food=food, **kwargs)

    def check_BFS(self):
        #Run BFS searching and return the full path of best way to food from BFS searching
        queue = [[self.snake.get_head()]]
        while queue:
            path = queue[0]
            future_head = path[-1]

            # If snake eats the food, return the next move after snake's head
            if future_head == self.food.location:
                return path

            for next_node in self._get_neighbors(future_head):
                if (
                    self.Check_Move(node=next_node, snake=self.snake)
                    or self.is_node_in_queue(node=next_node, queue=queue)
                ):
                    continue
                new_path = list(path)
                new_path.append(next_node)
                queue.append(new_path)

            queue.pop(0)

    def next_node(self):
        # return the path to the next move
        path = self.check_BFS()
        return path[1]

class Fowardcheck(Player):
    def __init__(self, snake: Snake, food: Food, **kwargs):
        super().__init__(snake=snake, food=food, **kwargs)
        self.kwargs = kwargs

    def run_forwardcheck(self):
        bfs = BFS(snake=self.snake, food=self.food, **self.kwargs)
        path = bfs.check_BFS()
        print("BFS")
        if path is None:
            snake_tail = Food()
            snake_tail.location = self.snake.body[0]
            snake = Snake(body=self.snake.body[1:])
            longest_path = Largest_path(snake=snake, food=snake_tail, **self.kwargs).run_longest()
            next_node = longest_path[0]
            print("BFS is not possible so look for head to tail")
            return next_node

        length = len(self.snake.body)
        Imaginary_body = (self.snake.body + path[1:])[-length:]
        Imaginary_tail = Food()
        Imaginary_tail.location = (self.snake.body + path[1:])[-length - 1]
        Imaginary_snake = Snake(body=Imaginary_body)
        Imaginary_SL = Largest_path(snake=Imaginary_snake, food=Imaginary_tail, **self.kwargs)
        Imaginary_l_path = Imaginary_SL.run_longest()
        if Imaginary_l_path is None:
            snake_tail = Food()
            snake_tail.location = self.snake.body[0]
            snake = Snake(body=self.snake.body[1:])
            longest_path = Largest_path(snake=snake, food=snake_tail, **self.kwargs).run_longest()
            next_node = longest_path[0]
            print("virtual snake not reachable, trying head to tail")
            # print(next_node)
            return next_node
        else:
            # print("BFS accepted")
            return path[1]

class Largest_path(BFS):
    # Taking BFS's path as an imput and chaning it to longest path
    def __init__(self, snake: Snake, food: Food, **kwargs):
        super().__init__(snake=snake, food=food, **kwargs)
        self.kwargs = kwargs

    def run_longest(self):
        # Check in each direction if it can be replaced with current move. IF there a replacement change it to longer move.
        path = self.check_BFS()
        if path is None:
            print(f"No path found")
            return

        i = 0
        while True:
            try:
                direction = self.node_sub(path[i], path[i + 1])
            except IndexError:
                break

            # Build a dummy snake with body and longest path for checking if node replacement is valid
            snake_path = Snake(body=self.snake.body + path[1:], **self.kwargs)

            # directions
            for neibhour in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if direction == neibhour:
                    x, y = neibhour
                    diff = (y, x) if x != 0 else (-y, x)

                    extra_node_1 = self.node_add(path[i], diff)
                    extra_node_2 = self.node_add(path[i + 1], diff)

                    if snake_path.dead_checking(head=extra_node_1) or snake_path.dead_checking(head=extra_node_2):
                        i += 1
                    else:
                        path[i + 1:i + 1] = [extra_node_1, extra_node_2]
                    break

        # snake head
        return path[1:]

class Astar(Player):
    def __init__(self, snake: Snake, food: Food, **kwargs):
        super().__init__(snake=snake, food=food, **kwargs)
        self.kwargs = kwargs

    def Implement_Astar(self):
        came_from = {}
        close_list = set()
        goal = self.food.location
        start = self.snake.get_head()
        dummy_snake = Snake(body=self.snake.body)
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        G_S = {start: 0}
        F_S = {start: heuristic(start, goal)}
        open_list = [(F_S[start], start)]
        print(start, goal, open_list)
        while open_list:
            current = min(open_list, key=lambda x: x[0])[1]
            open_list.pop(0)
            print(current)
            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                    print(data)
                return data[-1]

            close_list.add(current)

            for neighbor in neighbors:
                neighbor_node = self.node_add(current, neighbor)

                if dummy_snake.dead_checking(head=neighbor_node) or neighbor_node in close_list:
                    continue
                if sum(map(abs, self.node_sub(current, neighbor_node))) == 2:
                    diff = self.node_sub(current, neighbor_node)
                    if dummy_snake.dead_checking(head=self.node_add(neighbor_node, (0, diff[1]))
                                                 ) or self.node_add(neighbor_node, (0, diff[1])) in close_list:
                        continue
                    elif dummy_snake.dead_checking(head=self.node_add(neighbor_node, (diff[0], 0))
                                                   ) or self.node_add(neighbor_node, (diff[0], 0)) in close_list:
                        continue
                expected_g = G_S[current] + heuristic(current, neighbor_node)
                if expected_g < G_S.get(neighbor_node, 0) or neighbor_node not in [i[1] for i in open_list]:
                    G_S[neighbor_node] = expected_g
                    F_S[neighbor_node] = expected_g + heuristic(neighbor_node, goal)
                    open_list.append((F_S[neighbor_node], neighbor_node))
                    came_from[neighbor_node] = current

class Mixed(Player):
    def __init__(self, snake: Snake, food: Food, **kwargs):
        super().__init__(snake=snake, food=food, **kwargs)
        self.kwargs = kwargs

    def Getout(self):
        head = self.snake.get_head()
        largest_n_food_d = 0
        n_h = None
        for diff in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neibhour = self.node_add(head, diff)

            if self.snake.dead_checking(head=neibhour, check=True):
                continue

            n_food_d = (
                abs(neibhour[0] - self.food.location[0]) + abs(neibhour[1] - self.food.location[1])
            )
            # we are finding the bigger manhattan distance to food and then it has path to its tail
            if largest_n_food_d < n_food_d:
                snake_tail = Food()
                snake_tail.location = self.snake.body[1]
                snake = Snake(body=self.snake.body[2:] + [neibhour])
                bfs = BFS(snake=snake, food=snake_tail, **self.kwargs)
                path = bfs.check_BFS()
                if path is None:
                    continue
                largest_n_food_d = n_food_d
                n_h = neibhour
        return n_h

    def run_mixed(self):
        """
        Mixed strategy
        """
        bfs = BFS(snake=self.snake, food=self.food, **self.kwargs)

        path = bfs.check_BFS()
        # we will follow the tail to getout
        if path is None:
            return self.Getout()

        #Imaginary snake is checking if after eating the food path to tail is available.
        length = len(self.snake.body)
        Imaginary_body = (self.snake.body + path[1:])[-length:]
        Imaginary_tail = Food()
        Imaginary_tail.location = (self.snake.body + path[1:])[-length - 1]
        Imaginary_snake = Snake(body=Imaginary_body)
        Imaginary_SL = BFS(snake=Imaginary_snake, food=Imaginary_tail, **self.kwargs)
        Imaginary_l_path = Imaginary_SL.check_BFS()
        if Imaginary_l_path is None:
            return self.Getout()
        else:
            return path[1]





@dataclass
class SnakeGame(Base):
    fps: int = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

        pygame.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Game')

    def launch(self):
        while True:
            self.game()
            # self.showGameOverScreen()
            self.pause_game()

    def game(self):
        snake = Snake(**self.kwargs)

        food = Food(**self.kwargs)
        food.refresh(snake=snake)

        step_time = []


        while True:
            # AI Player
            for event in pygame.event.get():  # event handling loop
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.terminate()

            start_time = time.time()

            #BFS
            #new_head = BFS(snake=snake, food=food, **self.kwargs).next_node()

            #Astar 
            #new_head = Astar(snake=snake, food=food, **self.kwargs).Implement_Astar()

            #Forwardcheck
            #new_head = Fowardcheck(snake=snake, food=food, **self.kwargs).run_forwardcheck()

            #Mixed method 
            new_head = Mixed(snake=snake, food=food, **self.kwargs).run_mixed()
            end_time = time.time()
            move_time = end_time - start_time
            print(move_time)
            step_time.append(move_time)

            snake.move(new_head=new_head, food=food)

            if snake.is_dead:
                print(snake.body)
                print("Dead")
                break
            elif snake.eaten:
                food.refresh(snake=snake)

            if snake.score + snake.initial_length >= self.Block_w * self.Block_h:
                break

            self.display.fill(B)
            self.draw_panel()
            self.draw_snake(snake.body)

            self.draw_food(food.location)
            pygame.display.update()
            self.clock.tick(self.fps)

        print(f"Score: {snake.score}")
        print(f"Mean step time: {self.mean(step_time)}")

    @staticmethod
    def terminate():
        pygame.quit()
        sys.exit()

    def pause_game(self):
        while True:
            time.sleep(0.2)
            for event in pygame.event.get():  # event handling loop
                if event.type == QUIT:
                    self.terminate()
                if event.type == KEYUP:
                    if event.key == K_ESCAPE:
                        self.terminate()
                    else:
                        return

    def draw_snake(self, S_B):
        for snake_block_x, snake_block_y in S_B:
            x = snake_block_x * self.Block_size
            y = snake_block_y * self.Block_size
            snake_block = pygame.Rect(x, y, self.Block_size - 1, self.Block_size - 1)
            pygame.draw.rect(self.display, W, snake_block)

        # Draw snake's head
        x = S_B[-1][0] * self.Block_size
        y = S_B[-1][1] * self.Block_size
        snake_block = pygame.Rect(x, y, self.Block_size - 1, self.Block_size - 1)
        pygame.draw.rect(self.display, G, snake_block)

        # Draw snake's tail
        x = S_B[0][0] * self.Block_size
        y = S_B[0][1] * self.Block_size
        snake_block = pygame.Rect(x, y, self.Block_size - 1, self.Block_size - 1)
        pygame.draw.rect(self.display, B, snake_block)

    def draw_food(self, food_location):
        food_x, food_y = food_location
        food_block = pygame.Rect(food_x * self.Block_size, food_y * self.Block_size, self.Block_size, self.Block_size)
        pygame.draw.rect(self.display, R, food_block)

    def draw_panel(self):
        for x in range(0, self.window_width, self.Block_size):  # draw vertical lines
            pygame.draw.line(self.display, DG, (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.Block_size):  # draw horizontal lines
            pygame.draw.line(self.display, DG, (0, y), (self.window_width, y))


if __name__ == '__main__':
    SnakeGame().launch()
