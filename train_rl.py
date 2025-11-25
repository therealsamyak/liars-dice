"""Training script for RL agents in Liar's Dice."""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt

from parameters import (
    NUM_DICE_PER_PLAYER,
    RL_DISCOUNT_FACTOR,
    RL_EPSILON,
    RL_EPSILON_DECAY,
    RL_EPSILON_MIN,
    RL_LEARNING_RATE,
    RL_TRAINING_EPISODES,
)
from src.game_state import ObservableState
from src.opponent_policies import TruthTellingPolicy
from src.q_learning_agent import QLearningAgent
from src.rl_utils import compute_training_reward
from src.simulation import GameSimulation


def train_self_play(
    agent: QLearningAgent,
    num_episodes: int = 10000,
    save_interval: int = 1000,
    output_dir: str = "rl_models",
    verbose: bool = True,
):
    """Train RL agent using self-play.
    
    Args:
        agent: RL agent to train
        num_episodes: Number of training episodes
        save_interval: Save checkpoint every N episodes
        output_dir: Directory to save models
        verbose: Whether to print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    episode_lengths = []
    win_rates = []  # Rolling win rate
    window_size = 100  # Window for rolling statistics
    
    wins = 0
    
    for episode in range(1, num_episodes + 1):
        # Create simulation with two RL agents (self-play)
        # For simplicity, we train one agent playing against itself
        simulation = GameSimulation(
            num_dice_per_player=NUM_DICE_PER_PLAYER,
            player2_policy_type="truth_telling",  # Start with simpler opponent
        )
        
        # Override player 1 policy with RL agent
        from src.rl_policy import RLPolicy
        simulation.player1_policy = RLPolicy(agent)
        
        # Track episode transitions for learning
        episode_data = []
        
        # Run game with custom tracking
        starting_player = 1 if episode % 2 == 0 else 2
        winner, game_states = simulation.simulate_game(
            verbose=False, starting_player=starting_player
        )
        
        # Compute reward
        reward = compute_training_reward(winner, player_id=1)
        episode_rewards.append(reward)
        episode_lengths.append(len(game_states))
        
        if winner == 1:
            wins += 1
        
        # Note: For proper Q-Learning we'd need to track state-action pairs during game
        # For now, we'll implement a simpler training approach
        # This is a simplified version - would need full trajectory tracking
        
        # Decay epsilon
        agent.reset_episode()
        
        # Track rolling win rate
        if episode >= window_size:
            recent_rewards = episode_rewards[-window_size:]
            win_rate = sum(1 for r in recent_rewards if r > 0) / window_size
            win_rates.append(win_rate)
        
        # Print progress
        if verbose and episode % 100 == 0:
            recent_win_rate = wins / episode if episode > 0 else 0
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Win Rate: {recent_win_rate:.2%} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Q-Table Size: {len(agent.q_table)}"
            )
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(
                output_dir, f"q_agent_episode_{episode}.json"
            )
            agent.save(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(output_dir, "q_agent_final.json")
    agent.save(final_path)
    
    # Plot training curves
    plot_training_curves(
        episode_rewards, episode_lengths, win_rates, output_dir
    )
    
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Overall win rate: {wins / num_episodes:.2%}")
    
    return agent


def train_with_trajectory_tracking(
    agent: QLearningAgent,
    num_episodes: int = 10000,
    opponent_type: str = "truth_telling",
    save_interval: int = 1000,
    output_dir: str = "rl_models",
    verbose: bool = True,
):
    """Train RL agent with full trajectory tracking for proper Q-Learning.
    
    Args:
        agent: RL agent to train
        num_episodes: Number of training episodes
        opponent_type: Type of opponent ("truth_telling" or "optimal")
        save_interval: Save checkpoint every N episodes
        output_dir: Directory to save models
        verbose: Whether to print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Track metrics
    episode_rewards = []
    win_rates = []
    window_size = 100
    wins = 0
    
    # Get opponent policy
    if opponent_type == "truth_telling":
        opponent = TruthTellingPolicy(NUM_DICE_PER_PLAYER)
    else:
        # Could add optimal policy here
        opponent = TruthTellingPolicy(NUM_DICE_PER_PLAYER)
    
    for episode in range(1, num_episodes + 1):
        # Manual game loop for trajectory tracking
        from src.game_state import GameStatus, State
        from src.rules import check_win_condition
        import random
        
        # Initialize game
        state = State.initialize(NUM_DICE_PER_PLAYER)
        current_player = 1 if episode % 2 == 0 else 2
        
        # Track trajectory
        trajectory = []  # [(obs, action, reward, next_obs, done), ...]
        
        # Game loop
        while state.game_status == GameStatus.ACTIVE:
            # Create observable state for current player
            if current_player == 1:
                obs = ObservableState(
                    hand1=state.hand1.copy(),
                    bid_history=state.bid_history.copy(),
                    game_status=state.game_status,
                )
                
                # Agent selects action
                action = agent.get_action(obs)
                
                if action is None:
                    # Call LIAR
                    if state.bid_history:
                        last_bid = state.bid_history[-1]
                        winner = check_win_condition(state, caller=1, last_bid=last_bid)
                        state.game_status = GameStatus.GAME_OVER
                        
                        # Terminal reward
                        reward = compute_training_reward(winner, player_id=1)
                        trajectory.append((obs, action, reward, None, True))
                    break
                else:
                    # Make bid
                    state.bid_history.append(action)
                    
                    # Intermediate reward (0 for non-terminal)
                    next_obs = ObservableState(
                        hand1=state.hand1.copy(),
                        bid_history=state.bid_history.copy(),
                        game_status=state.game_status,
                    )
                    trajectory.append((obs, action, 0.0, next_obs, False))
                    
                    current_player = 2
            else:
                # Opponent's turn
                opponent_action = opponent.get_action(
                    state.hand2, state.bid_history
                )
                
                if opponent_action is None:
                    # Opponent calls LIAR
                    if state.bid_history:
                        last_bid = state.bid_history[-1]
                        winner = check_win_condition(state, caller=2, last_bid=last_bid)
                        state.game_status = GameStatus.GAME_OVER
                        
                        # Update last transition in trajectory with terminal reward
                        if trajectory:
                            obs, action, _, next_obs, _ = trajectory[-1]
                            reward = compute_training_reward(winner, player_id=1)
                            trajectory[-1] = (obs, action, reward, None, True)
                    break
                else:
                    state.bid_history.append(opponent_action)
                    current_player = 1
        
        # Learn from trajectory
        for obs, action, reward, next_obs, done in trajectory:
            agent.learn(obs, action, reward, next_obs, done)
        
        # Track metrics
        final_reward = trajectory[-1][2] if trajectory else 0.0
        episode_rewards.append(final_reward)
        
        if final_reward > 0:
            wins += 1
        
        # Decay epsilon
        agent.reset_episode()
        
        # Track rolling win rate
        if episode >= window_size:
            recent_rewards = episode_rewards[-window_size:]
            win_rate = sum(1 for r in recent_rewards if r > 0) / window_size
            win_rates.append(win_rate)
        
        # Print progress
        if verbose and episode % 100 == 0:
            recent_win_rate = (
                sum(episode_rewards[-window_size:]) / window_size 
                if episode >= window_size 
                else wins / episode
            )
            print(
                f"Episode {episode}/{num_episodes} | "
                f"Recent Win Rate: {(recent_win_rate + 1) / 2:.2%} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Q-Table: {len(agent.q_table)} states | "
                f"Updates: {agent.total_updates}"
            )
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(
                output_dir, f"q_agent_episode_{episode}.json"
            )
            agent.save(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(output_dir, "q_agent_final.json")
    agent.save(final_path)
    
    # Plot training curves
    plot_training_curves(episode_rewards, win_rates, output_dir)
    
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Overall win rate: {wins / num_episodes:.2%}")
    
    return agent


def plot_training_curves(episode_rewards, win_rates, output_dir):
    """Plot and save training curves.
    
    Args:
        episode_rewards: List of episode rewards
        win_rates: List of rolling win rates
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    axes[0].plot(episode_rewards, alpha=0.3, label="Episode Reward")
    # Smooth with moving average
    if len(episode_rewards) > 100:
        window = 100
        smoothed = [
            sum(episode_rewards[max(0, i - window):i + 1]) / min(window, i + 1)
            for i in range(len(episode_rewards))
        ]
        axes[0].plot(smoothed, label="Moving Average (100)", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Rewards Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot win rates
    if win_rates:
        axes[1].plot(win_rates, linewidth=2)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Win Rate")
        axes[1].set_title("Rolling Win Rate (100-episode window)")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label="50% baseline")
        axes[1].legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"training_curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training curves saved to {plot_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Liar's Dice")
    parser.add_argument(
        "--episodes",
        type=int,
        default=RL_TRAINING_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="truth_telling",
        choices=["truth_telling", "optimal"],
        help="Opponent type",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rl_models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print training progress",
    )
    
    args = parser.parse_args()
    
    # Create RL agent
    agent = QLearningAgent(
        num_dice_per_player=NUM_DICE_PER_PLAYER,
        player_id=1,
        learning_rate=RL_LEARNING_RATE,
        discount_factor=RL_DISCOUNT_FACTOR,
        epsilon=RL_EPSILON,
        epsilon_decay=RL_EPSILON_DECAY,
        epsilon_min=RL_EPSILON_MIN,
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}")
        agent.load(args.load_checkpoint)
    
    print("=" * 60)
    print("RL Agent Training for Liar's Dice")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Opponent: {args.opponent}")
    print(f"Learning Rate: {agent.alpha}")
    print(f"Discount Factor: {agent.gamma}")
    print(f"Initial Epsilon: {agent.epsilon}")
    print(f"Epsilon Decay: {agent.epsilon_decay}")
    print(f"Min Epsilon: {agent.epsilon_min}")
    print("=" * 60)
    
    # Train with trajectory tracking
    trained_agent = train_with_trajectory_tracking(
        agent=agent,
        num_episodes=args.episodes,
        opponent_type=args.opponent,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    # Print final stats
    print("\n" + "=" * 60)
    print("Training Complete - Final Statistics")
    print("=" * 60)
    stats = trained_agent.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
