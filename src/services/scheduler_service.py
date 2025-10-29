"""Scheduler service for automatic daily vehicle data fetching."""
import os
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from .vehicle_service import VehicleService
from .auth_service import AuthService


class SchedulerService:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.vehicle_service = VehicleService()
        self.auth_service = AuthService()
        self.is_running = False
        
    async def fetch_vehicle_data_job(self):
        """Scheduled job to fetch vehicle data at 5:30 AM daily."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"🕐 [SCHEDULED] Starting daily vehicle data fetch at {timestamp}")
            
            # Fetch live vehicle data
            vehicles_df = self.vehicle_service.get_live_vehicles()
            
            if not vehicles_df.empty:
                # Save to output directory with timestamp
                os.makedirs("output", exist_ok=True)
                output_file = f"output/vehicles_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                vehicles_df.to_csv(output_file, index=False)
                
                logger.success(f"✅ [SCHEDULED] Successfully fetched {len(vehicles_df)} vehicles and saved to {output_file}")
                
                # Log summary statistics
                active_vehicles = vehicles_df[vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])]
                logger.info(f"📊 [SCHEDULED] Active vehicles: {len(active_vehicles)}/{len(vehicles_df)}")
                
                # Log by ward if available
                if 'wardNo' in vehicles_df.columns:
                    ward_counts = vehicles_df['wardNo'].value_counts()
                    logger.info(f"🏘️ [SCHEDULED] Vehicles by ward: {dict(ward_counts)}")
                
            else:
                logger.warning(f"⚠️ [SCHEDULED] No vehicle data retrieved at {timestamp}")
                
        except Exception as e:
            logger.error(f"❌ [SCHEDULED] Daily vehicle fetch failed at {datetime.now()}: {e}")
    
    def start_scheduler(self):
        """Start the scheduler with daily 5:30 AM job."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        try:
            # Add daily job at 5:30 AM
            self.scheduler.add_job(
                self.fetch_vehicle_data_job,
                trigger=CronTrigger(hour=5, minute=30),
                id='daily_vehicle_fetch',
                name='Daily Vehicle Data Fetch',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            
            logger.success("🚀 Scheduler started - Daily vehicle fetch scheduled for 5:30 AM")
            logger.info("📅 Next run: " + str(self.scheduler.get_job('daily_vehicle_fetch').next_run_time))
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
            
        try:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("🛑 Scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
    
    def get_scheduler_status(self):
        """Get current scheduler status and job information."""
        if not self.is_running:
            return {
                "status": "stopped",
                "message": "Scheduler is not running"
            }
        
        try:
            job = self.scheduler.get_job('daily_vehicle_fetch')
            if job:
                return {
                    "status": "running",
                    "job_name": job.name,
                    "next_run_time": str(job.next_run_time),
                    "trigger": str(job.trigger),
                    "message": "Daily vehicle fetch scheduled for 5:30 AM"
                }
            else:
                return {
                    "status": "running",
                    "message": "Scheduler running but no jobs found"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting scheduler status: {e}"
            }
    
    async def trigger_manual_fetch(self):
        """Manually trigger the vehicle data fetch for testing."""
        logger.info("🔧 Manual vehicle data fetch triggered")
        await self.fetch_vehicle_data_job()
        return {"status": "completed", "message": "Manual fetch completed"}