#!/usr/bin/env python
from random import randint

from pydantic import BaseModel
from typing import List
from crewai.flow import Flow, listen, start, router

from eng_team_flow.crews.business.business_crew import BusinessCrew
from eng_team_flow.crews.manager.technical.technical_manager import TechnicalManagerCrew
from eng_team_flow.crews.engineering.software.software_engineer import (
    SoftwareEngineer
)


class EngTeamState(BaseModel):
    business_technical_refinement_counter: int = 0
    business_use_case: str = ""
    provide_more_clarity: bool = False
    clarifications: str = ""
    task_list: List[dict[str, str]] = None



class EngTeamFlow(Flow[EngTeamState]):
    
    
    @start(
        "provide_more_clarity"
    )
    def generate_business_usecase(self):
        if self.state.business_technical_refinement_counter > 2:
            print("Too many refinements")
            return "refinement_limit_exceeded"
        print("Generating business usecase.")
        result = (
            BusinessCrew()
            .crew()
            .kickoff(
                inputs={
                    "product_name": "Trader's Desk",
                    "competitor_product_name": "Robinhood",
                    "clarifications": self.state.clarifications
                }
            )
        )
        print(f"Business usecase generated: {result}")
        self.state.business_use_case = result["business_use_case"]

    @router(generate_business_usecase)
    def generate_engineering_task_list(self):
        print("Generating engineering task list.")
        result = (
            TechnicalManagerCrew()
            .crew()
            .kickoff(
                inputs={
                    "requirement": self.state.business_use_case
                }
            )
        )
        self.state.clarifications = result["clarification_query"]
        if result["provide_more_clarity"]:
            self.state.business_technical_refinement_counter += 1
            print(
                f"business refinement_counter now is: {self.state.business_technical_refinement_counter}"
            )
            return "provide_more_clarity"
        print(f"type -> result tasks: {type(result['tasks'])}")
        self.state.task_list = result["tasks"]
        return "initiate_code"
    
    @listen("initiate_code")
    def generate_code(self):
        final_task_details = "\n\n".join(
            [
                f"{task_detail.task_name} \n\n {task_detail.task_description}" for task_detail in self.state.task_list
            ]
        )
        final_task_details = f"{self.state.business_use_case}\n\n{final_task_details}"
        result = (
            SoftwareEngineer(
                output_file_name=f"agent_output/all_code.py"
            )
            .crew()
            .kickoff(
                inputs={
                    "task_details": final_task_details
                }
            )
        )        
    

def kickoff():
    eng_team_flow = EngTeamFlow()
    eng_team_flow.kickoff()


def plot():
    eng_team_flow = EngTeamFlow()
    eng_team_flow.plot()


if __name__ == "__main__":
    kickoff()
